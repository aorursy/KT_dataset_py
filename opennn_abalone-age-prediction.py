import pandas as pd

import numpy as np

import tensorflow as tf

from matplotlib import pyplot as plt

import seaborn as sns

import os

__DATA_DIR__ = os.path.join('..', 'input', 'abalone-dataset')

print(os.listdir(__DATA_DIR__))
__DATA_PATH__ = os.path.join(__DATA_DIR__, 'abalone.csv')

data = pd.read_csv(__DATA_PATH__)

data.head()
print('This dataset has {} observations with {} features'.format(data.shape[0], data.shape[1]))
data.describe()
_ = data.hist(figsize=(20, 10), grid = False, layout = (2, 4), bins = 30)
numerical_features = data.select_dtypes(include = [np.number]).columns

categorical_features = data.select_dtypes(include = [np.object]).columns
numerical_features
categorical_features
data.groupby('Sex').mean().sort_values('Rings')
_ = sns.pairplot(data[numerical_features])
_ = sns.heatmap(data[numerical_features].corr(), annot = True)
data = pd.get_dummies(data)

data.head()
X, y = data.drop('Rings', axis = 1), data['Rings']