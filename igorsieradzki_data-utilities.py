import os

import numpy as np

from typing import Sequence



data_dir = '../input'



def load_data(which: str):

    """

    Loads data from a csv file

    :param which: str

        Which data to load, train or test

    """

    assert which in ['train', 'test']

    

    if which == 'train':

        data = np.loadtxt(fname=os.path.join(data_dir, 'train_data.csv'), delimiter=',', skiprows=1)

        labels = np.loadtxt(fname=os.path.join(data_dir, 'train_labels.csv'), delimiter=',', skiprows=1)

        return data, labels

    elif which == 'test':

        data = np.loadtxt(fname=os.path.join(data_dir, 'test_data.csv'), delimiter=',', skiprows=1)

        return data

    

def save_prediction(prediction: Sequence[int], 

                    path: str = 'submission.csv'):

    """

    Saves a sequence of predictions into a csv file with additional index column

    :param prediction: Sequence of ints

        Predictions to save

    :param path: str

        Path to a file to save into

    """

    

    pred_with_id = np.stack([np.arange(len(prediction)), prediction], axis=1)

    np.savetxt(fname=path, X=pred_with_id, fmt='%d', delimiter=',', header='id,label', comments='')
train_data, train_labels = load_data(which='train')

print(train_data.shape)

print(train_labels.shape)
test_data = load_data(which='test')

print(test_data.shape)
pred = np.random.randint(0, 10, size=test_data.shape[0])

save_prediction(prediction=pred, path='random.csv')