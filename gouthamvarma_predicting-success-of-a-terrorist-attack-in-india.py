# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding = 'ISO-8859-1')

data = data[data['country_txt']=='India']
data.head()
data = data[data.describe().columns] #Ignoring the columns with strings
data = data[data['doubtterr']==0] #considering the data where terrorist attack took place for certain
data.head()
data = data[['imonth','iday','latitude', 'longitude','multiple', 'success', 'suicide',

       'attacktype1','targtype1',

       'natlty1', 'individual','weaptype1','nkill']]
data.isnull().sum()
data['latitude'].fillna(data['latitude'].mean(), inplace = True)

data['longitude'].fillna(data['longitude'].mean(), inplace = True)

data['natlty1'].fillna(data['natlty1'].mode()[0], inplace = True)

data['nkill'].fillna(data['nkill'].mode()[0], inplace = True)

data.isnull().sum()
fig, axis = plt.subplots(figsize = (20,20))

sns.heatmap(data.corr(), vmin = 0, cmap='Blues', annot = True, ax = axis)
sns.countplot(x = 'success', data = data)
X = data[['imonth','iday','latitude', 'longitude','multiple','suicide','attacktype1','targtype1','natlty1', 'individual','weaptype1','nkill']]

y = data['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state =1)
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy_score(y_test, prediction)