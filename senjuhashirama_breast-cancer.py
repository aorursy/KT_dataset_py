#importing necessary packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#reading .csv files and droppping unnecessary columns

data = pd.read_csv("../input/data.csv", index_col = "id")

data.drop(data.columns[-1], axis=1, inplace = True)

data.head()
#EDA

data.info()
#creating dummy columns

df = pd.get_dummies(data, drop_first= True)

df.head()
#Creating Feature and target data

x = df.drop(df.columns[-1], axis=1)

y = df["diagnosis_M"]

x.head()
#creating pipeline object

steps = [("scaler", StandardScaler()), ("tree", RandomForestClassifier())]

pipeline = Pipeline(steps)
#setting parameters for Cross_validation

k = list(range(1, 30))

k_mf = list(range(1,9))

parameters = {"tree__n_estimators": k,"tree__max_features": k_mf,

              "tree__min_samples_leaf": k_mf,"tree__criterion": ["gini", "entropy"]}
#Splitting data into train, test data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 21)

y_test.shape
#create Cross_validation object

cv = RandomizedSearchCV(pipeline,parameters, cv = 5)
#fitting the Model

cv.fit(x_train, y_train)
#Predicting Y_test and y_train

y_pred = cv.predict(x_test)

y_train_pred = cv.predict(x_train)
#calculating Accuracy Score on Train and Test data

a = accuracy_score(y_test, y_pred)

b = accuracy_score(y_train, y_train_pred)

print("Test_data_accuracy_score:",a,"\n" "Train_data_accuracy_score:",b)
#Printing Best parameters and best score

print("Tuned Logistic Regression Parameter: {}".format(cv.best_params_))

print("Tuned Logistic Regression Accuracy: {}".format(cv.best_score_))