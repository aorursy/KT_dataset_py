# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn import utils

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
hd = pd.read_csv("../input/heart.csv")

hd.head(20)
hd[hd.target == 0]
print(hd.dtypes)
hd.info()
pd.value_counts(hd['target'].values, sort=False)
sns.heatmap(hd.corr())
X = hd.iloc[:,:-1].values

y = hd.iloc[:,13].values



X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size = 0.2,random_state= 0)
#Logistic regression



LR = LogisticRegression()

LR.fit(X_train,y_train)
#Predicting for the entire dataset



y_pred = LR.predict(X_test)

#Accuracy of the model



score = LR.score(X_test,y_test)

print(score)
#Now we will apply confusion matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
#Let us apply KNN now





KNN = KNeighborsClassifier(n_neighbors=3)

KNN.fit(X_train,y_train)
y_pred = KNN.predict(X_test)
print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))