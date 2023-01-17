import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

import os

print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.shape
train.describe()
corrmat = train.corr(method='pearson')

f, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)
train.columns[train.isna().any()].tolist()
train.fillna(9999, inplace=True)

test.fillna(9999, inplace=True)
X = train.drop(["TARGET_5Yrs", "Id", "Name"], axis = 1)

Y = train["TARGET_5Yrs"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2222)
model = LogisticRegression(n_jobs=-1, random_state=10)
model.fit(X_train, y_train)
model.score(X_test, y_test)   
test_X = test.drop(["Name", "Id"], axis=1)
test_X = model.predict(test_X)
submission = pd.DataFrame({"Id":test["Id"],

                         "TARGET_5Yrs":test_X})
submission.head()
submission.to_csv("submission.csv", index=False)