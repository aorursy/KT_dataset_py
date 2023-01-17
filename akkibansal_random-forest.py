# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



data = pd.read_csv("../input/heart.csv")



'''Exploratory Data Analysis'''



# checking the first 5 rows of the data

data.head()

# checking the last 5 rows of the data

data.tail()

#check data

data.info() # no null values

data.describe()



#check correlation

fig, ax = plt.subplots(figsize=(10,10)) 

sns.set(font_scale=1.25)

hm = sns.heatmap(data.corr(), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})

plt.show()

#no correlation b/w variables found





'''Machine Learning'''

#Separate target variable

y=data.iloc[:,-1]

data=data.iloc[:,:-1]

#split the data into train and test using 80/20

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

#applying Random forest

RF = RandomForestClassifier()

RF.fit(X_train,y_train)

y_pred = RF.predict(X_test)

print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))