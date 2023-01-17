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
df = pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")
df.head()
df.columns
df.info()
df
df.isna().sum()
df.describe().T
df["Value"].median()
df["Value"].value_counts()
Methods_missing_data = (df["Methods"].isna().sum()/len(df["Methods"]))*100

Methods_missing_data
Notes_missing_data = (df["Notes"].isna().sum()/len(df["Notes"]))*100

Notes_missing_data
df = df.drop(columns=["Methods","Notes"])
df = df.dropna()
df.columns
df["Race/ Ethnicity"].value_counts()
df["Place"].value_counts()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
plt.figure(figsize=(13,10))

sns.scatterplot(x=df["Race/ Ethnicity"],y=df["Value"],hue=df["Gender"])
plt.figure(figsize=(13,10))

sns.barplot(x=df["Indicator Category"], y=df["Value"])
df.drop(columns=['Year','Indicator','BCHC Requested Methodology','Source'], inplace=True)

df.columns
df_categorical_col = df.select_dtypes(exclude=np.number).columns

df_categorical_col
df_numeric_col = df.select_dtypes(include=np.number).columns

df_numeric_col
df_onehot = pd.get_dummies(df[df_categorical_col])

df_onehot.head(5)
df_after_encoding = pd.concat([df[df_numeric_col],df_onehot], axis = 1)

df_after_encoding.head(5)
df_after_encoding.shape
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y = df_after_encoding["Value"]
x = df_after_encoding.drop(columns = "Value")
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=1)
model = LinearRegression()
model.fit(train_x,train_y)
print(model.intercept_)
print(model.coef_)
print("Predicting train data")

train_predict = model.predict(train_x)

print("Predicting test data")

test_predict = model.predict(test_x)

print(" ")

print("MAE")

print("Train data: ",mean_absolute_error(train_y,train_predict))

print("Test data: ",mean_absolute_error(test_y,test_predict))

print(" ")

print("MSE")

print("Train data: ",mean_squared_error(train_y,train_predict))

print("Test data: ",mean_squared_error(test_y,test_predict))

print(" ")

print("RMSE")

print("Train data: ",np.sqrt(mean_squared_error(train_y,train_predict)))

print("Test data: ",np.sqrt(mean_squared_error(test_y,test_predict)))

print(" ")

print("R^2")

print("Train data: ",r2_score(train_y,train_predict))

print("Test data: ",r2_score(test_y,test_predict))