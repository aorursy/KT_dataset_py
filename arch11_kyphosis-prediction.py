# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("/kaggle/input/kyphosis-dataset/kyphosis.csv")

df.shape
df.head()
df.isnull().sum()
dictionary = {"absent":0,"present":1}

Kyphosis = df["Kyphosis"].map(dictionary)

df["Kyphosis"] = Kyphosis
df
y = df["Kyphosis"]

X = df.drop("Kyphosis",axis=1)
sns.pairplot(df,hue="Kyphosis")
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn import model_selection

from catboost import CatBoostClassifier

from xgboost import XGBClassifier



log_reg = LogisticRegression()

rf_clf = RandomForestClassifier(n_estimators=500)

dt_clf = DecisionTreeClassifier()

svm_clf = SVC()

cat_clf  = CatBoostClassifier()

xgb =  XGBClassifier()

log_reg.fit(X,y)

rf_clf.fit(X,y)

dt_clf.fit(X,y)

svm_clf.fit(X,y)

cat_clf.fit(X,y)





print(model_selection.cross_val_score(log_reg,X,y,cv=5,scoring="roc_auc").mean())

print(model_selection.cross_val_score(rf_clf,X,y,cv=5,scoring="roc_auc").mean())

print(model_selection.cross_val_score(dt_clf,X,y,cv=5,scoring="roc_auc").mean())

print(model_selection.cross_val_score(svm_clf,X,y,cv=5,scoring="roc_auc").mean())



print(model_selection.cross_val_score(cat_clf,X,y,cv=5,scoring="roc_auc").mean())

print(model_selection.cross_val_score(xgb,X,y,cv=5,scoring="roc_auc").mean())