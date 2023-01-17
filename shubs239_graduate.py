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
df=pd.read_csv("../input/Admission_Predict.csv")
df.head()
import seaborn as sns
sns.distplot(df['GRE Score'],bins=40)
sns.pairplot(df)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
# Linear Regression
from sklearn.model_selection import train_test_split
X=df.drop('Chance of Admit ',axis=1)
y=df['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
predict_lr=lr.predict(X_test)
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_test,predict_lr)))
#Good approximation but let's checkout if we can further reduce it
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=500)
rfr.fit(X_train,y_train)
predict_rf=rfr.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,predict_rf)))
#Yes, we were able to reduce the error further