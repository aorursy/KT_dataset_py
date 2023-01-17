from sklearn import cross_validation

from sklearn import tree

from sklearn import svm

from sklearn import ensemble

from sklearn import neighbors

from sklearn import linear_model

from sklearn import metrics

from sklearn import preprocessing
%matplotlib inline 



from IPython.display import Image

import matplotlib as mlp

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import sklearn

import seaborn as sns

import tensorflow as tf
dataset = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')



print (dataset.shape)





X = dataset.iloc[:, dataset.columns!='phone number'].values

X = X[:,1:-1]

y = dataset.iloc[:, -1].values

print(X[0])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:, 2] = le.fit_transform(X[:, 2])

lw = LabelEncoder()

X[:, 3] = le.fit_transform(X[:, 3])

print(X[0])

Y = dataset["churn"].value_counts()

sns.barplot(Y.index, Y.values)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

print(X[0])
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=8, activation='relu'))

ann.add(tf.keras.layers.Dense(units=8, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 20, epochs = 100)

y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("The accuracy achieved is:",accuracy_score(y_test, y_pred)*100)
feature = 'state'

fig, ax = plt.subplots(1, 2, figsize=(20, 18))

dataset[dataset.churn == True][feature].value_counts().plot('bar', ax=ax[1]).set_title('State wise-Churned')

dataset[dataset.churn == False][feature].value_counts().plot('bar', ax=ax[0]).set_title('State wise-Not Churned')

feature = 'area code'

fig, ax = plt.subplots(1, 2, figsize=(20, 18))

dataset[dataset.churn == True][feature].value_counts().plot('bar', ax=ax[1]).set_title('Area code wise-Churned')

dataset[dataset.churn == False][feature].value_counts().plot('bar', ax=ax[0]).set_title('Area code wise-Not Churned')
numerical_features = ["total day charge","total eve charge","total night charge","total intl charge" ]

dataset[numerical_features].describe()
dataset[numerical_features].hist(bins=30, figsize=(20, 14))
