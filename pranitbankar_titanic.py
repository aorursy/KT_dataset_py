# -*- coding: utf-8 -*-

"""

Created on Tue Oct 29 18:37:36 2019

https://www.kaggle.com/c/titanic/overview

"""



import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder

import titanic_helper as H

import matplotlib.pyplot as plt



import os

os.chdir(r'/kaggle/working')



titanic_df = pd.read_csv("../input/titanic/train.csv")
titanic_df.set_index('PassengerId', inplace=True)



titanic_df.drop("Name",axis=1, inplace=True)

titanic_df.drop("Ticket",axis=1, inplace=True)



titanic_df.drop("Cabin",axis=1, inplace=True)



titanic_df.loc[(titanic_df.Sex == "male") & (pd.isna(titanic_df.Age)), "Age"] = 29.0

titanic_df.loc[(titanic_df.Sex == "female") & (pd.isna(titanic_df.Age)), "Age"] = 27.0



titanic_df = titanic_df[pd.notnull(titanic_df["Embarked"])]
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

titanic_df_enc = pd.DataFrame(enc.fit_transform(titanic_df[titanic_df.columns[titanic_df.dtypes==object].tolist()]))



titanic_df.drop("Sex",axis=1, inplace=True)

titanic_df.drop("Embarked",axis=1, inplace=True)



titanic_df_enc.index = titanic_df.index

titanic_df = pd.concat([titanic_df, titanic_df_enc], axis=1)



cols = titanic_df.columns.tolist()

cols[-5:] = ["Female", "Male", "Embarked-C", "Embarked-Q", "Embarked-S"]

titanic_df.columns = cols



y_train = np.array(titanic_df.iloc[:,0])

X_train = np.array(titanic_df.iloc[:,1:])
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
best_svm = H.best_svm()

best_svm.fit(X_train, y_train)
test_df = pd.read_csv("../input/titanic/test.csv")

test_df.set_index('PassengerId', inplace=True)



test_df.drop("Name",axis=1, inplace=True)

test_df.drop("Ticket",axis=1, inplace=True)



test_df.drop("Cabin",axis=1, inplace=True)



test_df.loc[(test_df.Sex == "male") & (pd.isna(test_df.Age)), "Age"] = 29.0

test_df.loc[(test_df.Sex == "female") & (pd.isna(test_df.Age)), "Age"] = 27.0



test_df = test_df[pd.notnull(test_df["Embarked"])]



test_df_enc = pd.DataFrame(enc.transform(test_df[test_df.columns[test_df.dtypes==object].tolist()]))



test_df.drop("Sex",axis=1, inplace=True)

test_df.drop("Embarked",axis=1, inplace=True)



test_df_enc.index = test_df.index

test_df = pd.concat([test_df, test_df_enc], axis=1)



cols = test_df.columns.tolist()

cols[-5:] = ["Female", "Male", "Embarked-C", "Embarked-Q", "Embarked-S"]

test_df.columns = cols



#y_test = np.array(test_df.iloc[:,0])

X_test = np.array(test_df.iloc[:,:])

X_test[152][4] = 7.895 #Fare is missing so replace with median for Male, 3rd class, Embarked from S without SibSp and Parch

X_test = sc.transform(X_test)
pred = best_svm.predict(X_test)





pass_id = np.array(test_df.index)

pass_id = np.reshape(pass_id, (pass_id.shape[0],1))

pred = np.reshape(pred, (pred.shape[0],1))

res = pd.DataFrame(np.concatenate((pass_id , pred ), axis = 1))

res.columns=["PassengerId", "Survived"]

res.set_index("PassengerId", inplace=True)

res.to_csv(r"outputSVM.csv")
from IPython.display import FileLink

FileLink(r"outputSVM.csv")