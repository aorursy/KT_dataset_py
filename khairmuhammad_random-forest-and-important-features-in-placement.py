# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

data.head()
print(data.shape)

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
data = data.fillna(0)

data.isna().sum()
print("before\n",data.dtypes)

for col in cat_cols:

    data[col] = data[col].astype("category")

print("after\n\n", data.dtypes)
le = LabelEncoder()

for cal in cat_cols:

    data[cal] = le.fit_transform(data[cal])

data.head()
X = data.drop(["sl_no","status","salary"], axis=1)

y = le.fit_transform(data["status"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("Decision Tree Accuracy", accuracy_score(y_test, pred) * 100,"%")

print("Importance of Features", clf.feature_importances_ * 100)
importance = clf.feature_importances_

tot=0

for i,v in enumerate(importance):

    print('Feature: %0s, Score: %.5f' % (X.columns[i],v))

    tot=tot+v

print(tot)



plt.xlabel('Feature')

plt.ylabel('Degree of Importance')

plt.bar(X.columns, importance)

plt.xticks(rotation=90)

plt.show()