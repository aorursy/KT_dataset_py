# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.head(50)
df.columns
df.isnull().sum()
df.corr()
df["Unnamed: 32"].isnull().sum()
df.shape
df.drop("Unnamed: 32",axis = 1, inplace = True)
df.columns
from sklearn.model_selection import train_test_split
X  = df
y = df.diagnosis
X.drop("diagnosis",axis = 1, inplace = True)
X,X_test,y,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X,y)
pred = lr.predict(X_test)
pred
from sklearn.metrics import accuracy_score

accuracy_score(pred,y_test)*100
from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier(random_state = 1)

dc.fit(X,y)
pred2 = dc.predict(X_test)
accuracy_score(pred2,y_test)*100
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 1,n_estimators = 100)
rf.fit(X,y)
pred3 = rf.predict(X_test)
accuracy_score(pred3,y_test)*100
from xgboost import XGBClassifier
xg = XGBClassifier(learning_rate = 0.001,n_estimators = 100)
xg.fit(X,y)
pred4 = xg.predict(X_test)
accuracy_score(pred4,y_test)*100