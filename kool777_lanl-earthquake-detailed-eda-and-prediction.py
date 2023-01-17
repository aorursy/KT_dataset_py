import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

import seaborn as sns

import xgboost as xgb
from os import listdir

print(listdir("../input"))
train = pd.read_csv("../input/train.csv", nrows=10000000,

                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

train.head(5)
print("Train: rows:{} cols:{}".format(train.shape[0], train.shape[1]))
fig, ax = plt.subplots(2,1, figsize=(20,12))

ax[0].plot(train.index.values, train.time_to_failure.values, c="darkred")

ax[0].set_title("Quaketime of 10 Mio rows")

ax[0].set_xlabel("Index")

ax[0].set_ylabel("Quaketime in ms");

ax[1].plot(train.index.values, train.acoustic_data.values, c="mediumseagreen")

ax[1].set_title("Signal of 10 Mio rows")

ax[1].set_xlabel("Index")

ax[1].set_ylabel("Acoustic Signal");