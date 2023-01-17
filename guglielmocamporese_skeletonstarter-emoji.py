##################################################

# Imports

##################################################



import numpy as np

import cv2

import os

import pandas as pd

import matplotlib.pyplot as plt

import emoji





##################################################

# Params

##################################################



DATA_BASE_FOLDER = '/kaggle/input/emojify-challenge'





##################################################

# Utils

##################################################



def label_to_emoji(label):

    """

    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed

    """

    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)
##################################################

# Load dataset

##################################################



df_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))

y_train = df_train['class']

df_validation = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))

y_validation = df_validation['class']

emoji_dictionary = {

    '0': '\u2764\uFE0F',

    '1': ':baseball:',

    '2': ':smile:',

    '3': ':disappointed:',

    '4': ':fork_and_knife:'

}



# See some data examples

print('EXAMPLES:\n####################')

for idx in range(10):

    print(f'{df_train["phrase"][idx]} -> {label_to_emoji(y_train[idx])}')
# Load phrase representation

x_train = np.load(

    os.path.join(DATA_BASE_FOLDER, 

                 'train.npy')).reshape(len(df_train), -1)

x_validation = np.load(

    os.path.join(DATA_BASE_FOLDER, 

                 'validation.npy')).reshape(len(df_validation), -1)

print(f'Word embedding size: {x_train.shape[-1]}')
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

x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy')).reshape(len(submission), -1)

if y_test_pred is not None:

    submission['class'] = y_test_pred

submission.to_csv('my_submission.csv', index=False)