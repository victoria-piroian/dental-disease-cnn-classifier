# Dental Image Classification - Oral Disease Diagnosis

**Spotting Oral Diseases from Dental Images Using Deep Learning**

---

## Overview
This project focuses on classifying common oral diseases from dental images using a combination of classical image processing techniques, feature extraction, and deep learning. The workflow includes:

- **Data preprocessing:** Noise reduction, contrast enhancement, color channel enhancement (gum color), and standardization.
- **Data augmentation:** Random rotations, flips, crops, blurring, noise addition, and color jitter to balance classes.
- **Dataset splitting:** Proportional stratified sampling for training, validation, and test sets.
- **Baseline modeling:** HOG feature extraction with SVM and hyperparameter tuning.
- **Primary modeling:** Convolutional Neural Networks (CNNs) with multiple convolutional layers, batch normalization, dropout, and optional deeper architectures.

---

## Project Structure

- `Data/Raw Images/` : Original dental images organized by disease categories.
- `notebooks/` : Jupyter notebooks with experiments, preprocessing, and model training.
- `DentalDiagnosisCNN.py` : Primary CNN model implementation.
- `outputs/` : Generated CSVs for training/validation loss/error and confusion matrices.
- `README.md` : Project overview and documentation.

---

## Key Features

1. **Image Preprocessing**
   - Noise reduction using Non-Local Means Denoising
   - Contrast mapping via CLAHE
   - Gum color enhancement
   - Standardization to reduce intensity variation

2. **Data Augmentation**
   - Random rotations, horizontal flips
   - Random resized crops
   - Gaussian blur, noise, and color jitter
   - Balanced dataset creation for oversampled classes

3. **Modeling**
   - **Baseline:** HOG feature extraction + SVM classifier
   - **CNN Primary Model:** Custom CNN with multiple convolutional layers, max-pooling, batch normalization, and dropout
   - **Deeper CNN Version:** Additional convolutional layer for improved feature extraction

4. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrices and normalized confusion matrices
   - Train/validation loss and error curves

---

## Usage

1. Clone the repository and mount your dataset:

```bash
git clone <repo-url>
```
2. Install dependencies:
```
pip install torch torchvision scikit-learn opencv-python matplotlib seaborn scikit-image
```
3. Set the dataset path and run preprocessing, baseline model, or CNN model training notebooks.
4. View outputs such as accuracy, classification reports, and confusion matrices.

## Goals
- Improve accessibility and accuracy in oral disease detection.
- Demonstrate the impact of preprocessing and augmentation on model performance.
- Compare classical ML methods (HOG + SVM) with deep learning approaches.

---

## Author
Victoria Piroian

University of Toronto

Faculty of Applied Science & Engineering, 2025
