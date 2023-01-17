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
import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
df = pd.read_csv("../input/h-1b-visa/h1b_kaggle.csv")
df.head()
new = df['WORKSITE'].str.split(",",n=2,expand = True)
df["Ciry"] = new[0]

df["State"] = new[1]

df.info()

df = df.drop(['WORKSITE'], axis = 1)

df.isnull().sum()
df['PREVAILING_WAGE'] = df['PREVAILING_WAGE'].fillna(df['PREVAILING_WAGE'].median())
df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].fillna('N')
df.isnull().sum()
df = df.dropna()
df.isnull().sum()
features = ['SOC_NAME','FULL_TIME_POSITION','PREVAILING_WAGE','State']

x = df[features]

y = df['CASE_STATUS']
x.head()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
x['SOC_NAME'] = LE.fit_transform(x['SOC_NAME'])

x['State'] = LE.fit_transform(x['State'])

x['FULL_TIME_POSITION'] = LE.fit_transform(x['FULL_TIME_POSITION'])
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state =0)
logreg = LogisticRegression()



logreg.fit(x_train, y_train)



y_pred = logreg.predict(x_test)



logreg.score(x_train, y_train)
#knn = KNeighborsClassifier(n_neighbors = 3)



#knn.fit(x_train, y_train)



#y_pred = knn.predict(x_test)



#knn.score(x_train, y_train)
#seed = 7

#scoring = 'accuracy'



# Spot Check Algorithms

#models = []

#models.append(('LR', LogisticRegression()))

#models.append(('LDA', LinearDiscriminantAnalysis()))

#models.append(('KNN', KNeighborsClassifier()))

#models.append(('CART', DecisionTreeClassifier()))

#models.append(('NB', GaussianNB()))

#models.append(('SVM', SVC()))

#print('MODEL ESTIMATED ACCURACY SCORES:\n')

#results = []

#names = []

#for x, model in models:

#	kfold = model_selection.KFold(n_splits=10, random_state=seed)

#	cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

#	results.append(cv_results)

#	x.append(x)

#	msg = "%s: %f (%f)" % (x, cv_results.mean(), cv_results.std())

#	print(msg)