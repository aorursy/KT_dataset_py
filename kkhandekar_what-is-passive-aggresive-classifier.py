#Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



#SK Learn Libraries

import sklearn

from sklearn.multiclass import OneVsRestClassifier   #1vs1 & 1vsRest Classifiers

from sklearn.linear_model import PassiveAggressiveClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import gc
#Load Data

url = '../input/iris/Iris.csv'

data = pd.read_csv(url, header='infer')

data.drop('Id',axis=1,inplace=True)
#Records

print("Total Records: ", data.shape[0])
#Records per Species

data.Species.value_counts()
#Stat Summary

data.describe().transpose()
#Inspect

data.head()
#Encoding Species columns (to numerical values)

data['Species'] = data['Species'].astype('category').cat.codes
#Feature & Target Selection

features = data.select_dtypes('float').columns

target = ['Species']



# Feature& Target  Dataset

X = data[features]

y = data[target]
#Split Parameters

test_size = 0.1



#Dataset Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0) 



#Reset Index

X_test = X_test.reset_index(drop=True)

y_test = y_test.reset_index(drop=True)
#PassiveAggresive Model

model = PassiveAggressiveClassifier(random_state=0, shuffle=True, loss= 'squared_hinge', average=True)

model.fit(X_train, y_train)

pa_pred = model.predict(X_test)



#Define 1-vs-Rest Strategy / Classifier

ovr = OneVsRestClassifier(model)



#fit model to training data

ovr.fit(X_train, y_train)



#Predications

ovr_pred = ovr.predict(X_test)



#Accuracy Score

acc = accuracy_score(y_test, ovr_pred)

acc_pa = accuracy_score(y_test,pa_pred)

print("Passive Aggresive Classifier Model Accuracy (with one-vs-Rest Strategy): ",'{:.2%}'.format(acc))

print("Passive Aggresive Classifier Model Accuracy (without one-vs-Rest Strategy): ",'{:.2%}'.format(acc_pa))