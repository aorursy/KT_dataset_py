# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# read csv and print first 5 lines
df = pd.read_csv("../input/multiTimeline.csv");

# rename columns to clear potential white space 
df.columns = ['month', 'diet', 'gym', 'finance']

# convert month as object to datetime so we can 
# eventually apply time series
df.month = pd.to_datetime(df.month)

# set month as index and set inplace equal to true 
# to modify the original index of dataframe
df.set_index('month', inplace=True)

# print the first five lines of the dataframe
df.head()

# plot time series for diet, gym and finance
df.plot(figsize=(20,10), linewidth = 5, fontsize=20)

# plot time series for only diet and gym
df[['diet', 'gym']].plot(figsize=(20,10), linewidth = 5, fontsize=20)

# store diet column in dataframe called diet
diet = df[['diet']]

# store gym column in dataframe called gym
gym = df[['gym']]


# https://campus.datacamp.com/courses/pandas-foundations/time-series-in-pandas?ex=9#skiponboarding
# plot the rolling average of the gym search term
gym.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)

# concatenate the rolling mean/moving average of diet and rolling mean of gym
df_rm = pd.concat([diet.rolling(12).mean(), gym.rolling(12).mean()], axis=1)

# plot concatenated rolling mean/moving average
df_rm.plot(figsize=(20,10), linewidth=5, fontsize=20)

# label x axis of plot 
plt.xlabel('Year', fontsize=20);

# import datasets
# 
from sklearn import datasets

# load the iris dataset
iris = datasets.load_iris()

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
# store the iris dataset in a dataframe
df_iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# list first 5 lines of iris dataframe 
print(df_iris.head())

# plot sepal length and sepal width from df_iris
sns.lmplot(x='sepal length (cm)', y='sepal width (cm)', fit_reg=True, data=df_iris, hue='target');

# plot petal length and petal width from df_iris
sns.lmplot(x='petal length (cm)', y='petal width (cm)', fit_reg=True, data=df_iris, hue='target');

print(df_iris.corr())
df_iris.groupby(['target']).corr()

df.plot(figsize=(20,10), linewidth=5, fontsize=20)

plt.xlabel('Year', fontsize=20);
#df_iris.head()
list(df_iris.columns.values)
