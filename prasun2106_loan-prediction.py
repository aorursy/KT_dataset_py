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
loan = pd.read_csv("../input/loan_borowwer_data.csv")
loan.head()
loan.info()
loan.describe()
loan.columns
loan["credit.policy"].dtype
loan.head()
loan["credit.policy"].value_counts()
loan.columns
sns.distplot(loan["int.rate"])
corr = loan.corr()
sns.heatmap(loan.corr(), cmap = "Greens")
plt.figure(figsize = (10,6))

loan["fico"].hist(color = "green", bins = 30)
plt.figure(figsize = (10,6))

loan[loan["credit.policy"]== 0]["fico"].hist(alpha = .5, color = "red", bins = 30, label = "credit.policy = 0 ")

loan[loan["credit.policy"]== 1]["fico"].hist(alpha = .5, color = "green", bins = 30, label = "credit.policy = 1 ")

plt.legend()
plt.figure(figsize = (10,6))

loan[loan["not.fully.paid"] == 1]["fico"].hist(alpha= 0.5, bins = 30, color = "green", label = "Not Fully Paid = 1")

loan[loan["not.fully.paid"] == 0]["fico"].hist(alpha= 0.5, bins = 30, color = "red", label = "Not Fully Paid = 0")

plt.legend()
loan[loan["purpose"] == "debt_consolidation"]["fico"].hist(alpha= 0.5, bins = 30, color = "green", label = "purpose = debt_consolidation")

loan[loan["purpose"] == "all_other"]["fico"].hist(alpha= 0.5, bins = 30, color = "red", label = "purpose = all_other")

loan[loan["purpose"] == "credit_card"]["fico"].hist(alpha= 0.5, bins = 30, color = "green", label = "purpose = credit_card")
loan["purpose"].value_counts()
plt.figure(figsize = (13,9))

sns.countplot(x= loan["purpose"], hue= "not.fully.paid", data = loan, palette= "Set2")
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
loan_copy = loan
loan_copy["purpose"] = lb.fit_transform(loan["purpose"])
loan_copy.head()
loan["purpose"] = loan_copy["purpose"]
loan.head()
#pd.get_dummies(loan["purpose"], drop_first = True)
from sklearn.model_selection import train_test_split
loan.columns
X= loan.drop("not.fully.paid", axis = 1)
y = loan["not.fully.paid"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state = 123)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
prediction_tree = dtree.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, prediction_tree)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
prediction_forest = rfc.predict(X_test)
metrics.accuracy_score(y_test, prediction_forest)