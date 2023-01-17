# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn import svm

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading Kaggle data to pandas dataframe 

wine = pd.read_csv("../input/winequality-red.csv")

wine.head(5)
#check for nulls in dataset

wine.isnull().sum()
#get more info about the dataset

wine.info()
#preprocessing data

bins = (2, 6.5, 8)

group_names = ['bad','good']

wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

wine['quality'].unique()
# assign value to newly created label

label_quality = LabelEncoder()

wine['quality'] = label_quality.fit_transform(wine['quality'])

wine.head(10)
#check the number of qulaity values

wine['quality'].value_counts()
#plot the quality

sns.countplot(wine['quality'])
#separate dataset as response variable and future variable

X = wine.drop('quality', axis =1)

y = wine['quality']
#Train and Test split data - keeping it simple : spliting data into two's

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Applying standard scaling to optimize results - most models require it for accuracy, our model has range of values (eg: chlorides, dioxides )



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



#Check our standard scaling values

X_train[0:10]
#Random Forest Classifier : 



rfc = RandomForestClassifier(n_estimators = 200)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)

pred_rfc[:20]
#how did the model perform?

#Classification : 82% correct

#Confusin Matrix : 263 correct, 10 wrong for bad wines - 22 correct, 25 wrong for good wines 

print(classification_report(y_test, pred_rfc))

print(confusion_matrix(y_test, pred_rfc))

#SVM Classifier

clf = svm.SVC()

clf.fit(X_train, y_train)

pred_clf = clf.predict(X_test)
#how did the model perform?

#Classification : 80% correct

#Confusin Matrix : 268 correct, 5 wrong for bad wines - 35 correct, 12 wrong for good wines 

print(classification_report(y_test, pred_clf))

print(confusion_matrix(y_test, pred_clf))
#Neural Networks - DEEP LEARNING with 500 iterations over the data

mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=500)

mlpc.fit(X_train, y_train)

pred_mlpc = mlpc.predict(X_test)

#how did the model perform?

#Classification : 76% correct

#Confusin Matrix : 257 correct, 16 wrong for bad wines - 23 correct, 24 wrong for good wines 

print(classification_report(y_test, pred_mlpc))

print(confusion_matrix(y_test, pred_mlpc))