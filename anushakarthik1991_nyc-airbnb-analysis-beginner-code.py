# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#loading data

data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()
#length of the dataset

len(data)
#display the shape of the dataset

tdata.shape
#overall description about data

data.describe().T
#basic information about dataset

data.info()
# this function used to display correlation between the variables

data.corr()
#check the datatypes in data

data.dtypes
#check for null values in data

data.isnull().sum()
#droping some columns

data.drop(['id', 'host_name', 'last_review', 'reviews_per_month'], axis = 1, inplace= True)
data.shape
#displays name of the columns present in data

data.columns
# displays unique values present in column

data.room_type.unique()
data.neighbourhood.unique()
data.neighbourhood_group.unique()
data.neighbourhood_group.value_counts()
#lets put the above one in visual format

sns.countplot(x='neighbourhood_group', data = data)

plt.title(' Popular Neighbourhood Group')

plt.show()
data.neighbourhood.value_counts()
data.room_type.value_counts()
sns.countplot(x='room_type', data= data)

plt.title('Room types view')

plt.figure(figsize=(12,8))

plt.show()
sns.countplot(x='room_type', hue='neighbourhood_group', data=data)

plt.title('Room types occupied by neighbourhood groups')

plt.figure(figsize=(12,6))

plt.show()
data1 = data[data.price < 500]

plt.figure(figsize=(10,8))

sns.violinplot(data=data1, x='neighbourhood_group', y='price')

plt.title('distribution of prices for each neighbourhood group')

plt.show()
plt.figure(figsize=(10,8))

sns.scatterplot(y='latitude', x='longitude', data=data, hue='neighbourhood_group', palette= 'hls')

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x='longitude', y='latitude', data=data, hue='availability_365', palette='coolwarm')

plt.show()