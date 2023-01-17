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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/Admission_Predict.csv')
df.head()
df.describe()
df = df.drop('Serial No.',axis = 1)
df.head()
sns.pairplot(df)
g = sns.heatmap(df.corr(),annot=True, cmap="PiYG")
g.figure.set_size_inches([8,8])
df['CGPA'].hist()
df['TOEFL Score'].hist()
df['GRE Score'].hist()
sns.boxplot(x='University Rating',y='Chance of Admit ',data=df)
X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit ']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.33, random_state=101)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
scaled_pred = lr.predict(X_test)
sns.regplot(x=y_test,y=scaled_pred,fit_reg=False)
from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test,scaled_pred))
df.columns
new_X = df[['GRE Score','TOEFL Score','University Rating','CGPA']]
new_scale = scaler.fit_transform(new_X)
X_train, X_test, y_train, y_test = train_test_split(new_scale, y, test_size=0.33, random_state=101)
lr.fit(X_train, y_train)
new_pred = lr.predict(X_test)
sns.regplot(x=y_test,y=new_pred,fit_reg=False)
print("MSE:", mean_squared_error(y_test,new_pred))
sns.distplot(scaled_pred)
plt.title('With All Columns')
sns.distplot(new_pred)
plt.title('GRE Score ,TOEFL Score , University Ratin , CGPA columns')