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
df = pd.read_csv("../input/insurance.csv")

df.head()
df.index
df.shape
df.info()
df.isna().sum()
df.corr()
df.region.unique()
df.smoker.unique()
df.replace({"yes":"1","no":"0"},inplace=True)
df.replace({"southwest":"0","southeast":"1","northwest":"2","northeast":"3"},inplace=True)
df.sex.unique()
df.replace({"female":"1","male":"0"},inplace=True)
df["sex"] = pd.to_numeric(df["sex"], errors='coerce')

df["smoker"] = pd.to_numeric(df["smoker"], errors='coerce')

df["region"] = pd.to_numeric(df["region"], errors='coerce')

df.info()
df['bmi'] = df['bmi'].astype('Int64')

df['expenses'] = df['expenses'].astype('Int64')

df.info()
df.corr()
x= df.drop(["expenses","sex","children","region"],axis=1)

x.head()
y=df["expenses"]

y.head()
import seaborn as sns

for i in x.columns:

    sns.pairplot(data=df,x_vars=i,y_vars="expenses")
g = sns.PairGrid(df, y_vars=["expenses"], x_vars=["smoker", "age","bmi"])

g.map(sns.regplot)
#x= x.drop("smoker",axis=1)

x= x.drop("age",axis=1)

#x= x.drop("bmi",axis=1)

x.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)
from sklearn.linear_model import LinearRegression

from sklearn import metrics

model = LinearRegression()

model.fit(x_train,y_train)  # Providing the training values to find model's intercept and slope
model.intercept_
model.coef_
y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)
train_aberror=metrics.mean_absolute_error(y_train,y_train_pred)

test_aberror=metrics.mean_absolute_error(y_test,y_test_pred)



train_sqerror=metrics.mean_squared_error(y_train,y_train_pred)

test_sqerror=metrics.mean_squared_error(y_test,y_test_pred)



train_sqlogerror=metrics.mean_squared_log_error(y_train,y_train_pred)

test_sqlogerror=metrics.mean_squared_log_error(y_test,y_test_pred)



train_r2Score=metrics.r2_score(y_train,y_train_pred)

test_r2Score=metrics.r2_score(y_test,y_test_pred)



print("mean_absolute_error train",train_aberror)

print("mean_absolute_error test",test_aberror)



print("mean_squared_error train",train_sqerror)

print("mean_squared_error test",test_sqerror)



print("mean_squared_log_error train",train_sqlogerror)

print("mean_squared_log_error test",test_sqlogerror)



print("r2_score train",train_r2Score)

print("r2_score test",test_r2Score)



train_rootsqerror=np.sqrt(train_sqerror)

test_rootsqerror=np.sqrt(test_sqerror)



print("squared root of mean_squared_error train",train_rootsqerror)

print("squared root of mean_squared_error test",test_rootsqerror)