# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv')

data.head()
data = data.drop(['Unnamed: 0', 'floor'], axis = 1)

data.head()
data.isnull().sum()
def remove_dollar(x):

    a = x[2:]

    result = ""

    for i in a:

        if i.isdigit() is True:

            result = result + i

        return result
data['hoa'] = pd.to_numeric(data['hoa'].apply(remove_dollar), errors = 'ignore')

data['rent amount'] = pd.to_numeric(data['rent amount'].apply(remove_dollar), errors = 'ignore' )

data['property tax'] = pd.to_numeric(data['property tax'].apply(remove_dollar), errors = 'ignore' )

data['fire insurance'] = pd.to_numeric(data['fire insurance'].apply(remove_dollar), errors = 'ignore' )

data['total'] = pd.to_numeric(data['total'].apply(remove_dollar), errors = 'ignore' )

data.head()
data.dtypes
data.head()
plt.figure(figsize = (5, 5))

sns.boxplot(data['total'])
#Removing Outliers

plt.figure(figsize = (7,7))

sns.set(style = "whitegrid")

f = sns.barplot(x = "rooms", y = "total", data = data)

f.set_title("Removing Outliers")

f.set_xlabel("No. of Rooms")

f.set_ylabel("Total Cost")
data.columns
columns = ['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'animal', 'furniture']

plt.figure(figsize = (30,30))

for i,var in enumerate(columns,1):

    plt.subplot(2,4,i)

    f = sns.barplot(x = data[var], y = data["total"])

    f.set_xlabel(var.upper())

    f.set_ylabel("Total Cost")
plt.figure(figsize=(12,12))

sns.heatmap(data.corr(), annot=True)