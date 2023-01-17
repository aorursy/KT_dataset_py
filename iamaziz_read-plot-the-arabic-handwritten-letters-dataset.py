from read_arabic_letters_dataset import (
    ReadArabicHandwrittenLettersDataset,
    plot_letter,
    plot_multiple_letters
)
data = ReadArabicHandwrittenLettersDataset()
data.x_train.shape, data.Y_train.shape, data.x_test.shape, data.Y_test.shape
type(data.x_train), type(data.Y_train)
plot_multiple_letters(data.x_train, 4, 17)
plot_letter(ix=7, data=data.x_train)
sample_data = data.get_sample_data([2, 6]) # two letters only e.g. h and b (each letter has 480 example in train and 120 in test)
plot_letter(0, sample_data)
plot_letter(600, sample_data)
plot_multiple_letters(sample_data, rows=4, cols=10)
%%writefile read_arabic_letters_dataset.py

"""This script reads and prepares the "ahcd1/Arabic Handwritten Characters Dataset CSV" dataset

By: https://github.com/iamaziz
Mon May 11 19:42:55 EDT 2020

Download the dataset at: https://www.kaggle.com/mloey1/ahcd1
"""
from functools import lru_cache
from typing import List

import numpy as np
import matplotlib.pyplot as plt  # for plotting letters


DATA_DIR = "/kaggle/input/ahcd1/"
FILE_NAMES = [
    "csvTrainImages 13440x1024.csv",
    "csvTestImages 3360x1024.csv",
    "csvTrainLabel 13440x1.csv",
    "csvTestLabel 3360x1.csv",
]


class ReadArabicHandwrittenLettersDataset:
    def __init__(self):
        self.files = [f"{DATA_DIR}{f}" for f in FILE_NAMES]
        # raw data
        print("reading data ..")
        self.X_train, self.X_test, self.Y_train, self.Y_test = [
            self.csv2ndarray(i) for i in range(len(self.files))
        ]
        print("raw: ")
        print(
            self.X_train.shape, self.X_test.shape, self.Y_train.shape, self.Y_test.shape
        )
        # pre-processed data
        self.x_train, self.x_test = self.reshape_data()
        print("reshaped:")
        print(
            self.x_train.shape, self.x_test.shape, self.Y_train.shape, self.Y_test.shape
        )

        self.LETTERS_IDS_train = list(set(self.Y_train))
        self.LETTERS_IDS_test = list(set(self.Y_test))

    @lru_cache()
    def csv2ndarray(self, i) -> np.ndarray:
        """Read csv matrix data with Numpy"""
        return np.genfromtxt(self.files[i], delimiter=",")

    def reshape_data(self):
        x_train = self.X_train.reshape((13440, 32, 32))
        x_test = self.X_test.reshape((3360, 32, 32))
        return x_train, x_test

    @staticmethod
    def indices_of_letter(letter_num: float, data: np.ndarray) -> List:
        return np.where(data == letter_num)[0]

    def _letters_indices_in_dataset(self, letters):
        # .. from train data
        sample_train_idxs = [self.indices_of_letter(l, self.Y_train) for l in letters]
        sample_train_idxs = [item for sublist in sample_train_idxs for item in sublist]
        # in train data each letter has 480 samples
        assert len(sample_train_idxs) == 480 * len(letters)

        # .. from test data
        sample_test_idxs = [self.indices_of_letter(l, self.Y_test) for l in letters]
        sample_test_idxs = [item for sublist in sample_test_idxs for item in sublist]
        # in test data each letter has 120 samples
        assert len(sample_test_idxs) == 120 * len(letters)

        print(len(sample_train_idxs), len(sample_test_idxs))

        # concat indices
        sample_idx = sample_train_idxs + sample_test_idxs
        len(sample_idx)

        return sample_train_idxs, sample_test_idxs

    def get_sample_data(self, sample_letters: List[int]) -> np.ndarray:
        train_idxs, test_idxs = self._letters_indices_in_dataset(letters=sample_letters)
        # select target letters data
        sample_train_data = self.x_train[train_idxs]
        sample_test_data = self.x_test[test_idxs]
        print(sample_train_data.shape, sample_test_data.shape)
        sample_data = np.concatenate([sample_train_data, sample_test_data])
        print(sample_data.shape)
        return sample_data


#################
# -- HELPERS -- #
#################


def plot_letter(ix: int, data: np.ndarray):
    number = data[ix].T
    plt.imshow(number, cmap="gray")
    plt.show()


def plot_multiple_letters(data: np.ndarray, rows=2, cols=3):
    """plot randomly selected letters from `data`"""
    from random import randint

    fig, axes = plt.subplots(rows, cols, figsize=(13, 7))

    idx = [(i, j) for i in range(rows) for j in range(cols)]
    for i, ix in enumerate(idx):
        ax = axes[ix]
        i = randint(0, len(data))
        d = data[i].T  # TODO: FIX letters reshaping and rotation! must be without .T
        ax.imshow(d, cmap="gray")
    plt.show()