import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

import numpy as np

%matplotlib inline
data_bdwd = pd.read_csv('../input/bangladesh-weather-dataset/Temp_and_rain.csv')
data_bdwd
data_bdwd.head(8)
data_bdwd.columns
data_bdwd.describe()
print('Total Entries: ', str(len(data_bdwd)))
data_bdwd['tem'].hist().plot()
data_bdwd['rain'].hist().plot()
data_bdwd.plot('Year', 'tem')
data_bdwd.plot('Year', 'rain')
print(max(data_bdwd['tem']))
print(min(data_bdwd['tem']))
data_bdwd.loc[data_bdwd.loc[:, 'tem'] == 29.526, :]

data_bdwd.loc[data_bdwd.loc[:, 'tem'] == 16.8006, :]
print(max(data_bdwd['rain']))
data_bdwd.loc[data_bdwd.loc[:, 'rain'] == 1012.02, :]
print(min(data_bdwd['rain']))
data_bdwd.loc[data_bdwd.loc[:, 'rain'] == 0.0, :]
sns.countplot(x = 'tem', data = data_bdwd)
sns.countplot(x = 'rain', data = data_bdwd)
data_bdwd.info()
data_bdwd.isnull()
data_bdwd.isnull().sum()
sns.heatmap(data_bdwd.isnull(), cmap = 'viridis')
data_bdwd.plot.scatter(x = 'Year', y = 'rain', c = 'tem', colormap='viridis')
data_bdwd.plot.scatter(x = 'Year', y = 'rain', c ='red')
data_bdwd.plot.scatter(x = 'Year', y = 'tem', c ='blue')