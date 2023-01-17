# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
info = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
info
#perc =[.20, .40, .60, .80] 
#desc = info.describe(percentiles = perc, include = ['float', 'int']) 
#desc
#import matplotlib.pyplot as plt
#for i in info.columns:
#    plt.figure()
#    plt.hist(info[i])
#    plt.suptitle(i, fontsize=20)

info.drop(['EmployeeNumber','EmployeeCount','Over18','StandardHours'], inplace=True,axis=1)
columns = info.columns
num_columns = info._get_numeric_data().columns
catlist = list(set(columns)-set(num_columns))
catlist
catlist.remove('Attrition')
tempcat = pd.concat([pd.get_dummies(info[col],col) for col in catlist], axis=1)
print(tempcat.columns)
tempcat
liklist = ['Education','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','PerformanceRating','RelationshipSatisfaction','WorkLifeBalance','JobLevel','StockOptionLevel']
templik = pd.concat([pd.get_dummies(info[col],col) for col in liklist], axis=1)
print(templik.columns)
templik
info['PerformanceRating'].plot.hist()
info.drop(liklist,axis=1,inplace=True)
info.drop(catlist,axis=1,inplace=True)
newinfo = pd.concat([info['Attrition'],tempcat,templik],axis=1)
info.drop('Attrition', axis=1, inplace=True)
info
info1 = pd.DataFrame(columns=info.columns)
for i in info.columns:
    info1[i] = pd.qcut(info[i], 5, labels=False,duplicates='drop')
info1
newinfo = pd.concat([info1, newinfo],axis=1)
newinfo
newinfo['Attrition'].replace({"Yes":1,"No":0}, inplace=True)
newinfo
x = newinfo.drop(['Attrition'],axis=1)
y = newinfo['Attrition']
x
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
score = accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
score
y_probpred = model.predict_proba(x_test)
probs = []
for term in y_probpred:
    probs.append(term[1])
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,probs)
from sklearn.feature_selection import RFE
selector = RFE(LogisticRegression(), n_features_to_select = 5)
selector = selector.fit(x_train,y_train)
y_predRFE = selector.predict(x_test)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
score = accuracy_score(y_test,y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("general accuracy" + str(score))
print("precision" + str(precision))
print("recall" + str(recall))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print ('coefficients',selector.estimator_.coef_)
selector.support_
x_train.columns[[15,29,47,51,69]]
x_train
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
nn = keras.Sequential()
nn.add(keras.Input(78))
nn.add(layers.Dense(20, activation="relu"))
nn.add(layers.Dense(20, activation="relu"))
nn.add(layers.Dense(1, activation='sigmoid'))
nn.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
nn.fit(x_train,y_train,epochs=4,validation_data=(x_test,y_test))
nn.fit(x_train,y_train,epochs=4,validation_data=(x_test,y_test))
nn.fit(x_train,y_train,epochs=4,validation_data=(x_test,y_test))
predictions = nn.predict(x_test)
predictions = predictions.flatten()
newlist = []
for term in predictions:
    if term > 0.5:
        newlist.append(1)
    else:
        newlist.append(0)

newlist
score = accuracy_score(y_test,newlist)
precision = precision_score(y_test, newlist)
recall = recall_score(y_test, newlist)
print("general accuracy" + str(score))
print("precision" + str(precision))
print("recall" + str(recall))
print(classification_report(y_test,newlist))
info2 = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
import matplotlib.pyplot as plt
for i in info2.columns:
    plt.figure()
    plt.hist(info2[i])
    plt.suptitle(i, fontsize=20)

import statsmodels.api as sm
data = info2[['JobRole','Attrition']]
print(sm.stats.Table.from_data(data).table_orig)
data = info2[['OverTime','Attrition']]
print(sm.stats.Table.from_data(data).table_orig)
data = info2[['JobLevel','Attrition']]
print(sm.stats.Table.from_data(data).table_orig)
data = info2[['EnvironmentSatisfaction','Attrition']]
print(sm.stats.Table.from_data(data).table_orig)
data = info2[['JobInvolvement','Attrition']]
print(sm.stats.Table.from_data(data).table_orig)
150/882
