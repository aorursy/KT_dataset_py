# Library

import kagglegym

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

%matplotlib inline

# Setting for gym and plot

p = sns.color_palette()

env = kagglegym.make()

observation = env.reset()
# Load data

train = observation.train

# Fill NaN(adhoc)

train_median = train.median()

train.fillna(train_median, inplace = True)

# Sampling to reduce calculation

train = train.sample(frac=0.1)
# Split features and y(...and remove id and timestamp for simplicity)

features = train[train.columns.difference(['id', 'timestamp', 'y'])]

y = train.y.values
# Lean by random dorest 

rf = RandomForestRegressor(n_jobs=-1)

rf.fit(features, y)
# Get importance by importance order

importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

print(features.columns.values[indices])

print(importances[indices])