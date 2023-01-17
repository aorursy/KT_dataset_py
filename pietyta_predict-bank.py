import numpy as np

import pandas as pd

import torch

import matplotlib.pyplot as plt

from torch import nn, optim

from torch.utils.data import DataLoader

import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

%matplotlib inline
data_train = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')

data = data_train.append(X_test, ignore_index=True, sort=False)

data = pd.get_dummies(data, dummy_na=True, drop_first=True)

data.drop('Id', axis=1, inplace=True)

data.isnull().values.any()