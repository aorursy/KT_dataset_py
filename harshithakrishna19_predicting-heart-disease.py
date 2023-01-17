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
# -*- coding: utf-8 -*-

"""

Created on Thu Nov  7 14:58:16 2019



@author: harshitha.k

"""

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



#read the csv(data) file

data_frame=pd.read_csv("dataset.csv")

#prints 5 rows of data from the csv

print(data_frame.head())

#prints the column names

print(data_frame.columns)

#prints information of the data frame

print(data_frame.info())

#prints the description of the data

print(data_frame.describe())

#check whether the data is balanced or not

print(data_frame.target.value_counts())

#THE VALUE IS IN RANGE OF 120-170 SO IT CAN BE CONSIDERED AS BALANCED DATASET

#print the shape of dataframe which shows rows and columns

print(data_frame.shape)

#Step-2:FEATURE SELECTION

#plt.figure(figsize=(10,10))

#sns.heatmap(data_frame.corr(),annot=True,fmt='.1f')

#plt.show()

corr=data_frame.corr()

#print(plt.figure(figsize=(10,10)))

#plt.subplots(figsize=(20,15))

print(sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns,annot=True,linewidths=1))

print(data_frame.hist())

dataset = pd.get_dummies(data_frame, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])



standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

print(dataset.head())

y = dataset['target']

X = dataset.drop(['target'], axis = 1)



randomforest_classifier= RandomForestClassifier(n_estimators=10)



score=cross_val_score(randomforest_classifier,X,y,cv=10)

print(score.mean())

#plt.matshow(data_frame.corr())

#plt.show()


































































