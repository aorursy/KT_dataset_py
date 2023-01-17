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
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop("customerID", 1, inplace=True)
text_cols = df.select_dtypes("object").columns
df[text_cols] = df[text_cols].astype("category")
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].astype("float")

df.info()
df.isnull().sum()
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)
df.isnull().sum()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
train_cols = df.columns[:-1]
target_col = "Churn"
test_ratio = 0.25
div = int(len(df)*test_ratio)
train = df[div:]
test = df[:div]
lr = LogisticRegression()
scores = []
accuracy = []
numeric_cols = df.select_dtypes("number").columns
for col in numeric_cols:
    lr.fit(np.array(train[col]).reshape(-1,1), train[target_col])
    scores.append(lr.score(np.array(test[col]).reshape(-1,1), test[target_col]))
    yhat = lr.predict(np.array(test[col]).reshape(-1,1))
    accuracy.append(accuracy_score(yhat, test[target_col]))
    
print(numeric_cols)
print(scores)
scores = []
accuracy = []
numeric_cols = df.select_dtypes("number").columns

lr.fit(train[numeric_cols], train[target_col])
scores.append(lr.score(test[numeric_cols], test[target_col]))
yhat = lr.predict(test[numeric_cols])
accuracy.append(accuracy_score(yhat, test[target_col]))
    
print(numeric_cols)
print(scores)
from sklearn.svm import SVC
svc = SVC()
svc.fit(train[numeric_cols], train[target_col])
svc.score(test[numeric_cols], test[target_col])
