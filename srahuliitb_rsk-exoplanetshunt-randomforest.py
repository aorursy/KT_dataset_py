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
train_df = pd.read_csv('/kaggle/input/kepler-labelled-time-series-data/exoTrain.csv')

test_df = pd.read_csv('/kaggle/input/kepler-labelled-time-series-data/exoTest.csv')
train_df.head()
test_df.head()
train_df.shape
train_df.isnull().sum()
train_df.isna().sum()
count_miss_values = 0

for column in train_df.columns:

    for item in train_df[column].isnull():

        if item == True:

            count_miss_values += 1

            

count_miss_values
# Importing Libraries

import matplotlib.pyplot as plt

import seaborn as sns
# First Star In The Dataset

star0 = train_df.iloc[0, :]

star0.head()
# Scatter Plot For First Star

plt.figure(figsize=(15, 5))

plt.scatter(pd.Series([i for i in range(1, len(star0))]), star0[1:])

plt.ylabel('Flux')

plt.show()
# Line Plot For First Star

plt.figure(figsize=(15, 5))

plt.plot(pd.Series([i for i in range(1, len(star0))]), star0[1:])

plt.ylabel('Flux')

plt.show()
# Second Star

star1 = train_df.iloc[1, :]

star1.head()
# Scatter Plot For Second Star

plt.figure(figsize=(15, 5))

plt.scatter(pd.Series([i for i in range(1, len(star1))]), star1[1:])

plt.ylabel('Flux')

plt.show()
# Line Plot For Second Star

plt.figure(figsize=(15, 5))

plt.plot(pd.Series([i for i in range(1, len(star1))]), star1[1:])

plt.ylabel('Flux')

plt.show()
train_df.tail()
# Last Star

star5086 = train_df.iloc[5086, :]

star5086.head()
# Scatter Plot For Last Star

plt.figure(figsize=(15, 5))

plt.scatter(pd.Series([i for i in range(1, len(star5086))]), star5086[1:])

plt.ylabel('Flux')

plt.show()
# Line Plot For Last Star

plt.figure(figsize=(15, 5))

plt.plot(pd.Series([i for i in range(1, len(star5086))]), star5086[1:])

plt.ylabel('Flux')

plt.show()
# Second-Last Star

star5085 = train_df.iloc[5085, :]

star5085.head()
# Scatter Plot For Second-Last Star

plt.figure(figsize=(15, 5))

plt.scatter(pd.Series([i for i in range(1, len(star5085))]), star5085[1:])

plt.ylabel('Flux')

plt.show()
# Line Plot For Second-Last Star

plt.figure(figsize=(15, 5))

plt.plot(pd.Series([i for i in range(1, len(star5085))]), star5085[1:])

plt.ylabel('Flux')

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report
# Split the dataframe into feature variables and the target variable.

x_train = train_df.iloc[:, 1:]

x_train.head()
y_train = train_df.iloc[:, 0]

y_train.head()
rf_clf1 = RandomForestClassifier(n_jobs=-1)

rf_clf1.fit(x_train, y_train)

rf_clf1.score(x_train, y_train)
x_test = test_df.iloc[:, 1:]

x_test.head()
y_test = test_df.iloc[:, 0]

y_test.head()
y_test.shape
y_predicted = rf_clf1.predict(x_test)

y_predicted.shape
# Confusion Matrix

# In binary classification, the count of true negatives is C(0, 0), false negatives is C(1, 0), true positives is C(1, 1) and false positives is C(0, 1).

# [[TP, FN], 

#  [FP, TN]]

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_predicted)
y_predicted = pd.Series(y_predicted)

y_predicted.value_counts()
y_test.value_counts()
print(classification_report(y_test, y_predicted))
accuracy_score(y_test, y_predicted)