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
sns.set()
data=pd.read_csv("/kaggle/input/kc-housesales-data/kc_house_data.csv")
data.head(5)
data.describe(include="all")
data.isna().sum()
r=data.columns
for i in r:
    print("'",i,"'has these many uniques",data[i].nunique())
data=data.drop(["id"],axis=1)
X=data.drop(["price"],axis=1)
y=data["price"]
ax = sns.scatterplot(x="price", y="date", data=data)
X=X.drop(["date"],axis=1)
X.dtypes
ax = sns.heatmap(data.corr())
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.35,random_state=0)
lr=LinearRegression(fit_intercept=True)
model=lr.fit(xtrain,ytrain)
prediction=lr.predict(xtest)
print("Train_Accuracy")
print(lr.score(xtrain,ytrain))
print("Test_Accuracy")
print(lr.score(xtest,ytest))
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_leaf=4,min_samples_split=10,random_state=0)
model=regressor.fit(xtrain, ytrain)
y_pred = regressor.predict(xtest)
print("Train_Accuracy")
print(regressor.score(xtrain,ytrain))
print("Test_Accuracy")
print(regressor.score(xtest,ytest))