import numpy as np

import pandas as pd

import tensorflow as tf

import keras



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set(palette="Set2")



from sklearn.model_selection import train_test_split

from sklearn.metrics import (accuracy_score, f1_score,average_precision_score, confusion_matrix,

                             average_precision_score, precision_score, recall_score, roc_auc_score, )

from mlxtend.plotting import plot_confusion_matrix



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler





from xgboost import XGBClassifier, plot_importance

from imblearn.over_sampling import SMOTE
# read dataset

dataset = pd.read_csv("../input/Churn_Modelling.csv")
# first five row of the dataset

dataset.head()
dataset.describe()
# checking datatypes and null values

dataset.info()
dataset.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)
_, ax = plt.subplots(1, 3, figsize=(18, 6))

plt.subplots_adjust(wspace=0.3)

sns.countplot(x = "NumOfProducts", hue="Exited", data = dataset, ax= ax[0])

sns.countplot(x = "HasCrCard", hue="Exited", data = dataset, ax = ax[1])

sns.countplot(x = "IsActiveMember", hue="Exited", data = dataset, ax = ax[2])
_, ax = plt.subplots(1, 3, figsize=(18, 6))

plt.subplots_adjust(wspace=0.3)

sns.swarmplot(x = "NumOfProducts", y = "Age", hue="Exited", data = dataset, ax= ax[0])

sns.swarmplot(x = "HasCrCard", y = "Age", data = dataset, hue="Exited", ax = ax[1])

sns.swarmplot(x = "IsActiveMember", y = "Age", hue="Exited", data = dataset, ax = ax[2])
encoder = LabelEncoder()

dataset["Geography"] = encoder.fit_transform(dataset["Geography"])

dataset["Gender"] = encoder.fit_transform(dataset["Gender"])
dataset["Age"].value_counts().plot.bar(figsize=(20,6))
facet = sns.FacetGrid(dataset, hue="Exited",aspect=3)

facet.map(sns.kdeplot,"Age",shade= True)

facet.set(xlim=(0, dataset["Age"].max()))

facet.add_legend()



plt.show()
_, ax =  plt.subplots(1, 2, figsize=(15, 7))

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = dataset, ax=ax[0])

sns.scatterplot(x = "Age", y = "CreditScore", hue = "Exited", cmap = cmap, sizes = (10, 200), data = dataset, ax=ax[1])
plt.figure(figsize=(8, 8))

sns.swarmplot(x = "HasCrCard", y = "Age", data = dataset, hue="Exited")
facet = sns.FacetGrid(dataset, hue="Exited",aspect=3)

facet.map(sns.kdeplot,"Balance",shade= True)

facet.set(xlim=(0, dataset["Balance"].max()))

facet.add_legend()



plt.show()
_, ax = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(x = "Balance", y = "Age", data = dataset, hue="Exited", ax = ax[0])

sns.scatterplot(x = "Balance", y = "CreditScore", data = dataset, hue="Exited", ax = ax[1])
facet = sns.FacetGrid(dataset, hue="Exited",aspect=3)

facet.map(sns.kdeplot,"CreditScore",shade= True)

facet.set(xlim=(0, dataset["CreditScore"].max()))

facet.add_legend()



plt.show()
plt.figure(figsize=(12,6))

bplot = dataset.boxplot(patch_artist=True)

plt.xticks(rotation=90)       

plt.show()
plt.subplots(figsize=(11,8))

sns.heatmap(dataset.corr(), annot=True, cmap="RdYlBu")

plt.show()
X = dataset.drop("Exited", axis=1)

y = dataset["Exited"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = GaussianNB()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy_score(pred, y_test)
clf = LogisticRegression()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy_score(pred, y_test)
clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy_score(pred, y_test)
clf = RandomForestClassifier(n_estimators = 200, random_state=200)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy_score(pred, y_test)
clf  = XGBClassifier(max_depth = 10,random_state = 10, n_estimators=220, eval_metric = 'auc', min_child_weight = 3,

                    colsample_bytree = 0.75, subsample= 0.9)



clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy_score(pred, y_test)
scaler = MinMaxScaler() 



bumpy_features = ["CreditScore", "Age", "Balance",'EstimatedSalary']



df_scaled = pd.DataFrame(data = X)

df_scaled[bumpy_features] = scaler.fit_transform(X[bumpy_features])
df_scaled.head()
X = df_scaled

sm  = SMOTE(random_state=42)

X_res, y_res = sm.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size= 0.2, random_state=7)
clf = XGBClassifier(max_depth = 12,random_state=7, n_estimators=100, eval_metric = 'auc', min_child_weight = 3,

                    colsample_bytree = 0.75, subsample= 0.8)

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1:", f1_score(y_test, y_pred))

print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))
# Confusion Matrix

confusion_matrix(y_test, y_pred)