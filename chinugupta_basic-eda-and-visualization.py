# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy.random as nr

import scipy.stats as stats

import sklearn.preprocessing as skpe

import sklearn.model_selection as ms

import sklearn.metrics as sklm

import sklearn.ensemble as ensemble

import sklearn.linear_model as lm

import sklearn.tree as tree

import sklearn.linear_model as lm

import sklearn.kernel_ridge as ridge

import xgboost as xgb

import lightgbm as lgb

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

import sklearn.pipeline as pipeline

from sklearn.compose import ColumnTransformer
path = "../input/craigslist-carstrucks-data/vehicles.csv"

data = pd.read_csv(path)

print(data.shape)

data.head()
data.describe()
data.info()
drop_cols = ['id', 'url', 'region', 'region_url', 'model', 'title_status', 'vin', 'size', 'image_url', 'description', 'county', 'state', 'lat', 'long']

data = data.drop(columns=drop_cols)
data.info()
# Let's look at the target variable first

data['price'].describe()
plt.figure(figsize=(3,6))

sns.boxplot(y='price', data=data, showfliers=False)
sns.distplot(data['year'])
plt.figure(figsize=(3,6))

sns.boxplot(y='year', data=data, showfliers=False)
plt.figure(figsize=(3,6))

sns.boxplot(y='odometer', data=data, showfliers=False)
sns.boxplot(y='price', x='fuel', data=data, showfliers=False)
plt.figure(figsize=(20,15))

plt.xticks(rotation=90)

sns.boxplot(y='price', x='manufacturer', data=data, showfliers=False)
plt.xticks(rotation=90)

sns.boxplot(y='price', x='cylinders', data=data, showfliers=False)
plt.xticks(rotation=90)

sns.boxplot(y='price', x='condition', data=data, showfliers=False)
sns.boxplot(y='price', x='transmission', data=data, showfliers=False)
sns.boxplot(y='price', x='drive', data=data, showfliers=False)
plt.xticks(rotation=90)

sns.boxplot(y='price', x='type', data=data, showfliers=False)
plt.xticks(rotation=90)

sns.boxplot(y='price', x='paint_color', data=data, showfliers=False)
# Getting data afterward 1980

data = data.loc[lambda data : data['year'] >= 1980]



plt.figure(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x='year', data=data)
print('Top 10 car manufacturing years: ')

print(data['year'].value_counts().iloc[:10])
plt.figure(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x='paint_color', data=data)
print('Top 10 paint colors preferred: ')

print(data['paint_color'].value_counts().iloc[:10])
plt.figure(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x='type', data=data)
print('Top 10 car types manufactured: ')

print(data['type'].value_counts().iloc[:10])
plt.figure(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x='fuel', data=data)
print('Top 5 car fuel types: ')

print(data['fuel'].value_counts().iloc[:5])
plt.figure(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x='drive', data=data)
plt.figure(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x='condition', data=data)
print('Top 5 car conditions: ')

print(data['condition'].value_counts().iloc[:5])
plt.figure(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x='cylinders', data=data)
print('Top 5 car cylinders type: ')

print(data['cylinders'].value_counts().iloc[:5])
plt.figure(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x='manufacturer', data=data)
print('Top 5 car manufacturers: ')

print(data['manufacturer'].value_counts().iloc[:5])
plt.figure(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x='transmission', data=data)
print('Top 3 car transmission: ')

print(data['transmission'].value_counts().iloc[:3])