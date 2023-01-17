##################################################

# Imports

##################################################



import numpy as np

import cv2

import os

import pandas as pd

import matplotlib.pyplot as plt





##################################################

# Params

##################################################



DATA_BASE_FOLDER = '/kaggle/input/image-classification-fashion-mnist'
##################################################

# Load dataset

##################################################



x_train = np.load(os.path.join(DATA_BASE_FOLDER, 'train.npy'))

x_valid = np.load(os.path.join(DATA_BASE_FOLDER, 'validation.npy'))

x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy'))

y_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))['class'].values

y_valid = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))['class'].values

y_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# Plot random images of different classes

plt.figure(figsize=(25, 5))

for idx in range(20):

    plt.subplot(1, 20, idx + 1)

    img = x_train[idx].reshape(28, 28)

    plt.title(f'{y_labels[y_train[idx]]}')

    plt.imshow(img, cmap='gray')

    plt.axis('off')

plt.show()
##################################################

# Process the data here, if needed

##################################################



'''

Any manipulation of the dataset in order to feed the data to the algorithm in the correct "format".

'''





















##################################################

# Implement you model here

##################################################

















##################################################

# Evaluate the model here

##################################################



# Use this function to evaluate your model

def accuracy(y_pred, y_true):

    '''

    input y_pred: ndarray of shape (N,)

    input y_true: ndarray of shape (N,)

    '''

    return (1.0 * (y_pred == y_true)).mean()



# Report the accuracy in the train and validation sets.















##################################################

# Save your test prediction in y_test_pred

##################################################



y_test_pred = None



# Create submission

submission = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'sample_submission.csv'))

if y_test_pred is not None:

    submission['class'] = y_test_pred

submission.to_csv('my_submission.csv', index=False)