import numpy as np

import pandas as pd

import gc

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode() 
import os

print(os.listdir("../input/ashrae-energy-prediction/"))
%%time

root = '../input/ashrae-energy-prediction/'

train_df = pd.read_csv(root + 'train.csv')

weather_train_df = pd.read_csv(root + 'weather_train.csv')

test_df = pd.read_csv(root + 'test.csv')

weather_test_df = pd.read_csv(root + 'weather_test.csv')

building_meta_df = pd.read_csv(root + 'building_metadata.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')
print('Size of train_df data', train_df.shape)

print('Size of weather_train_df data', weather_train_df.shape)

print('Size of weather_test_df data', weather_test_df.shape)

print('Size of building_meta_df data', building_meta_df.shape)
train_df.head()
weather_train_df.head()
weather_test_df.head()
building_meta_df.head()
train_df['meter'].value_counts()
# Want to see meter reading by meter type

train_df.groupby('meter')['meter_reading'].describe()
# checking missing data

total = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)

missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing__train_data.head(4)
# checking missing data

total = weather_train_df.isnull().sum().sort_values(ascending = False)

percent = (weather_train_df.isnull().sum()/weather_train_df.isnull().count()*100).sort_values(ascending = False)

missing_weather_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_weather_data.head(9)
# checking missing data

total = weather_test_df.isnull().sum().sort_values(ascending = False)

percent = (weather_test_df.isnull().sum()/weather_test_df.isnull().count()*100).sort_values(ascending = False)

missing_weather_test_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_weather_test_data.head(9)
# checking missing data

total = building_meta_df.isnull().sum().sort_values(ascending = False)

percent = (building_meta_df.isnull().sum()/building_meta_df.isnull().count()*100).sort_values(ascending = False)

missing_building_meta_df  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_building_meta_df.head(6)
# Number of each type of column

train_df.dtypes.value_counts()
weather_train_df.dtypes.value_counts()
building_meta_df.dtypes.value_counts()
# Number of unique classes in each object column

train_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Find correlations with the target and sort

correlations = train_df.corr()['meter_reading'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))