#cell of import libs

import numpy as np

import pandas as pd

from sklearn.svm import LinearSVC

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from sklearn import svm

data = pd.read_csv('datasets/train.csv')

results = pd.read_csv('datasets/test.csv')
results.head()
columns_target = ['Survived']

columns_train = ['Age','Pclass','Sex','Fare']
x = data[columns_train]

y = data[columns_target]



x_results = results[columns_train]
x['Age']=x['Age'].fillna(x['Age'].median())
x['Age'].isnull().sum()
d = {"male":0,

    "female":1}

x['Sex'] = x['Sex'].apply(lambda x:d[x])
X_train ,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.33,random_state=42)
clf = svm.LinearSVC()
clf.fit(X_train,Y_train)
clf.predict(X_test)

clf.score(X_test,Y_test)
teste = results

teste['Age']=teste['Age'].fillna(teste['Age'].median())

print(teste['Age'].isnull().sum())



teste1['Fare']=teste1['Fare'].fillna(teste1['Fare'].median())

print(teste1['Fare'].isnull().sum())
d = {"male":0,

    "female":1}

teste['Sex'] = teste['Sex'].apply(lambda x:d[x])
teste1 = teste[columns_train]
teste1.head()
resultado = clf.predict(teste1)
results['Survived'] = resultado
results =results.drop('Embarked', axis=1)

results.head()
results.to_csv(r'datasets\resultado4.csv', index = False)