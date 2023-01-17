import dask.dataframe as dd

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

import copy

import math

from sklearn.model_selection import cross_val_score
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_X = dd.read_csv("/kaggle/input/richters-predictor-modeling-earthquake-damage/train_values.csv")

train_y = dd.read_csv("/kaggle/input/richters-predictor-modeling-earthquake-damage/train_labels.csv")



df = train_X.merge(train_y, how="inner", on = "building_id")

sample = df.sample(frac=1, random_state=12).compute()
object_list = list(sample.select_dtypes("object").columns)

print(object_list)
colToDrop = []

for col in sample.columns:

    if "secondary" in col:

        colToDrop.append(col)

sample.drop(colToDrop, axis=1, inplace=True)
sns.countplot(x='damage_grade', data=sample)
sample = pd.get_dummies(sample)
X = sample.loc[:,sample.columns != "damage_grade"].values

y = sample.loc[:,sample.columns == "damage_grade"].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import GradientBoostingClassifier

boost = GradientBoostingClassifier(learning_rate=0.15,max_depth=5, min_samples_split=1200, n_estimators= 300, verbose = 1)

boost.fit(X_train, y_train.flatten())

y_pred = boost.predict(X_test)

from sklearn.metrics import f1_score

f1_score = f1_score(y_test, y_pred, average='micro')

print(f1_score)
from sklearn.metrics import confusion_matrix

conf_matriz = confusion_matrix(y_test, y_pred)

print(conf_matriz)
test_x = pd.read_csv("/kaggle/input/richters-predictor-modeling-earthquake-damage/test_values.csv")

test_data = pd.get_dummies(test_x)
for elem in test_data.columns:

  if elem not in sample.columns:

      test_data.drop(elem, axis=1, inplace=True)   
y_pred_test = boost.predict(test_data)

y_pred_test = pd.DataFrame(y_pred_test, columns = ["damage_grade"])

building = test_x.loc[:,"building_id"]

building = pd.DataFrame(building, columns = ["building_id"])

solucion = pd.concat([building, y_pred_test], axis = 1)



#solucion.to_csv("/kaggle/output/solution.csv", index = False)