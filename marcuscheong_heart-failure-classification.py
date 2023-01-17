#Data Manipulation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Data Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-whitegrid')

#OS interaction

import os
path = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"

heartDF = pd.read_csv(path)

heartDF.head()
#Check if there are null values

heartDF.isnull().sum()
from sklearn.tree import DecisionTreeClassifier,export_graphviz

from graphviz import render,Source

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



y = heartDF['DEATH_EVENT']

X = heartDF.drop(columns=['DEATH_EVENT'])



accuracy_dict = {}



x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

dtc = DecisionTreeClassifier(random_state=0)

dtc.fit(x_train,y_train)

y_pred = dtc.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

accuracy_dict["DecisionTreeClassifier"] = accuracy

print("Accuracy:",accuracy)

dotfile = open("dtc.dot", 'w')

export_graphviz(dtc, out_file = dotfile, feature_names = x_train.columns)

dotfile.close()
print("Decision path for Decision Tree Classifier")

render('dot', 'png', 'dtc.dot')

Source.from_file("dtc.dot")
from sklearn.feature_selection import VarianceThreshold

varDF = pd.DataFrame(X.var(),columns=['Feature Variance'])

varDF
thres = VarianceThreshold(threshold=0.2)

high_var = thres.fit_transform(X)

print("Columns deleted:",X.columns[~thres.get_support()])

x_train = x_train[ X.columns[(thres.get_support())]]

x_test = x_test[X.columns[(thres.get_support())]]

pd.DataFrame(high_var,columns= X.columns[(thres.get_support())])
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)



accuracy_dict["RandomForestClassifier"] = accuracy
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(x_train,y_train)

y_pred = xgb.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)



accuracy_dict["XGBoost Classifier"] = accuracy
from sklearn.ensemble import AdaBoostClassifier



abc = AdaBoostClassifier(n_estimators=50)

abc.fit(x_train,y_train)

y_pred = abc.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)



accuracy_dict["AdaBoost Classifier"] = accuracy
accuracy_dict
for k,v in accuracy_dict.items():

    if(v == max(accuracy_dict.values())):

        model_chosen = k



if model_chosen == "DecisionTreeClassifier":

    model = dtc

elif model_chosen == "RandomForestClassifier":

    model = rfc

elif model_chosen == "XGBoost Classifier":

    model = xgb

else:

    model = ada
print("Best model for this classification:", model_chosen)

print("Accuracy:",accuracy_dict[model_chosen])