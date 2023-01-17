



import numpy as np 

import pandas as pd 

df = pd.read_csv('/kaggle/input/titanic/train.csv') # importing the data

print(df.head())

df = df.dropna(subset=['Embarked'])

df['Age'] = df['Age'].fillna(df['Age'].median())

print(df.isnull().sum())

y = df['Survived'] # label space

x_num = df[['Pclass','Age',  'SibSp', 'Parch', 'Fare']] # numerical and oridinal values within the feature space

x_cat = df[['Sex', 'Embarked']]

x_cat = pd.get_dummies(x_cat)

X = pd.concat([x_num, x_cat], axis=1, sort=False)

X
# Fitting the model



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE

import seaborn as sns

import matplotlib.pyplot as plt





X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1)  # Spliting the data

X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)   # Preprocessing the data

X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)

LR = LogisticRegression(solver='saga')

LR.fit(X_train, y_train)

yhat = LR.predict(X_test)

print("Logistic regression accuracy:", metrics.accuracy_score(y_test, yhat)) #Finding out the accuracy



cm = metrics.confusion_matrix(y_test, yhat)

df_cm = pd.DataFrame(cm, range(2), range(2))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, annot=True)



sel = RFE(LR,5,step=1)

sel.fit(X_train, y_train)

sel.support_

X_train_lasso = sel.fit_transform(X_train, y_train)

X_test_lasso = sel.transform(X_test)

LR_rfe = LogisticRegression(solver='saga')

LR_rfe.fit(X_train_lasso, y_train)

score_lasso = LR_rfe.score(X_test_lasso, y_test)

print(score_lasso)
#SVM



from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1)  # Spliting the data

X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)   # Preprocessing the data

X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)

mdl = SVC(C=10, gamma='auto')

mdl.fit(X_train, y_train)

yhat_svm = mdl.predict(X_test)

print(metrics.accuracy_score(y_test, yhat_svm))



df2 = pd.read_csv('/kaggle/input/titanic/test.csv')

df2['Age'] = df2['Age'].fillna(df['Age'].median())

df2['Fare'] = df2['Fare'].fillna(df['Fare'].median())

x_num1 = df2[['Pclass','Age', 'SibSp', 'Parch', 'Fare']]

x_cat1 = df2[['Sex', 'Embarked']]

x_cat1 = pd.get_dummies(x_cat1)

X1 = pd.concat([x_num1, x_cat1], axis=1, sort=False)



ypred = LR.predict(X1)

pid = df2['PassengerId']

ypred = pd.DataFrame(ypred)

y1 = pd.concat([pid, ypred], axis=1, sort=False)

y1.to_csv('file.csv')