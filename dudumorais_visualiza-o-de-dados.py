from scipy.io.arff import loadarff

import pandas as pd

import matplotlib.pyplot as plt

import os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
os.listdir('/kaggle/input/')
path_train = '/kaggle/input//bankTR.arff'

path_teste = '/kaggle/input/bankTE.arff'



row_data_train = loadarff(path_train)

train_df_data = pd.DataFrame(row_data_train[0])

row_data_test = loadarff(path_teste)

test_df_data = pd.DataFrame(row_data_test[0])



display(train_df_data.dtypes)

train_df_data.head()
train_df_data.describe()
train_df_data.hist(figsize=(10,8))
# Analisar a probabilidade de sobrevivência pelo Sexo

train_df_data[['age', 'duration']].groupby(['age']).mean()
fig, (axis1) = plt.subplots(1,1, figsize=(30,4))

sns.barplot(x='age', y='duration', data=train_df_data, ax=axis1)
columns=['age', 'balance', 'day', 'duration']

pd.plotting.scatter_matrix(train_df_data[columns], figsize=(15, 10));