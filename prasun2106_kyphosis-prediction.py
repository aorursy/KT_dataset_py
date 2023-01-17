# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/kyphosis.csv")
data.head()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data["Kyphosis"] = labelencoder.fit_transform(data["Kyphosis"])
data.head()
data.describe()
sns.distplot(data["Age"])
sns.distplot(data["Number"])
sns.distplot(data["Start"])
data.describe()
sns.boxplot(x = "Kyphosis", y= "Age", data= data)
sns.boxplot(x = "Kyphosis", y= "Number", data= data, palette = "Greens")
plt.boxplot(data["Number"])
pd.DataFrame(data["Number"]).describe()
sns.boxplot(x = "Kyphosis", y= "Start", data= data, palette = "Greens")
sns.pairplot(data, hue = "Kyphosis", palette = "Greens")
sns.heatmap(data.corr(), cmap = "BuGn")
from sklearn.model_selection import train_test_split
X= data.drop("Kyphosis", axis = 1)
y = data["Kyphosis"]
X_train,X_test, y_train, y_test = train_test_split(X, y)
from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)
tree_pred = model_tree.predict(X_test

                               )
from sklearn import metrics
print(metrics.classification_report(y_test, tree_pred))
print(metrics.accuracy_score(y_test, tree_pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators= 200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
metrics.accuracy_score(y_test, rfc_pred)
print(metrics.classification_report(y_test, rfc_pred))
from sklearn import metrics

print(metrics.confusion_matrix(y_test, rfc_pred))
print(metrics.confusion_matrix(y_test, tree_pred))