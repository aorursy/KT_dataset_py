# importing modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina'
# reading training data.

data_train = pd.read_csv("/kaggle/input/santa-workshop-tour-2019/family_data.csv")
print(data_train.info())
data_train.head(10)
print("Column name - Unique values")

for col in data_train.columns:

    print(col, " - ", len(data_train[col].value_counts()))
# Visualize 'choice_0'

data_train['choice_0'].hist(figsize=(10, 4),bins=100);
# Visualize 'choice_1'

data_train['choice_1'].hist(figsize=(10, 4),bins=100);
# Visualize 'choice_2'

data_train['choice_2'].hist(figsize=(10, 4),bins=100);
# Visualize all choice features

features = data_train.columns[1:-1]



for feature in features:

    data_train[feature].hist(figsize=(70, 50),bins=100);
# Visualizing 'n_people' column

data_train['n_people'].hist(figsize=(10, 4),bins=7);