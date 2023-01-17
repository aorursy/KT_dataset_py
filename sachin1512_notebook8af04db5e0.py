import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
fname = '../input/breast-cancer.csv'

data = pd.read_csv(fname)

data.head()
data = data.drop('id',axis=1)
Y = data['diagnosis']

data = data.drop('diagnosis',axis=1)

data = data.drop('Unnamed: 32',axis=1)
def one_hot_encoder(x):

    if x=='M':

        return 1

    else:

        if x=='B':

            return 0

Y = Y.apply(one_hot_encoder)
def normalizer(df,key):

    xmean = df.mean()

    xmin = df.min()

    xmax = df.max()

    

    df = (df - xmean)/(xmax - xmin)

    

    return df



cols = data.columns

for i in cols:

    data[i] = normalizer(data[i],i)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import sklearn.linear_model

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
def nan_finder(df):

    col = df.columns

    nan_list = []

    for i in col:

        k = df[i].isnull().sum()

        nan_list.append([i,k])

        

    #print(nan_cols)

    for i in nan_list:

        print(i)
y = Y.values

x = data.values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)
model1 = XGBClassifier()

model1.fit(x_train,y_train)
y_pred1 = model1.predict(x_test)

predictions1 = [round(value) for value in y_pred1]

accuracy1 = accuracy_score(y_test, predictions1)

print("Accuracy: %.2f%%" % (accuracy1 * 100.0))
model2 = SVC()

model2.fit(x_train,y_train)
y_pred2 = model2.predict(x_test)

predictions2 = [round(value) for value in y_pred2]

accuracy2 = accuracy_score(y_test, predictions1)

print("Accuracy: %.2f%%" % (accuracy2 * 100.0))
model3 = sklearn.linear_model.LogisticRegression()

y_pred3 = model3.fit(x_train,y_train)
y_pred3 = model3.predict(x_test)

predictions3 = [round(value) for value in y_pred3]

accuracy3 = accuracy_score(y_test, predictions3)

print("Accuracy: %.2f%%" % (accuracy3 * 100.0))