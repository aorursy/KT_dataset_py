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

# importing libraries

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score


df = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv',parse_dates=["date"]) # Changed Time dtypes 
df.head()
df.info()
df["Day"] = df.date.dt.day
df["Month"] = df.date.dt.month 
df["Year"] = df.date.dt.year 
df.drop(["date"],axis="columns",inplace = True) 
df.head()
df.count()
df.isnull().sum()
hs_n = df["houses_sold"].mean()
n_c = df["no_of_crimes"].mean()
df1 = df.fillna({"houses_sold" : hs_n, "no_of_crimes" : n_c})
df1.head()
area = LabelEncoder()
code = LabelEncoder()

df1["area_n"] = area.fit_transform(df1["area"])
df1["code_n"] = code.fit_transform(df1["code"])
df1.drop(["area","code"],axis="columns",inplace=True)
df1.head()
no_house_sold = df1.groupby("Year")["average_price"].mean().reset_index()

plt.figure(figsize=[20,10])
plt.xticks(rotation=90)
plt.grid()
plt.xlabel("Year",fontsize=20,color = "g")
plt.ylabel("average_price",fontsize=20,color="g")
plt.title("Prices Per Year",color="magenta",fontsize=20)
sns.barplot("Year","average_price",data = no_house_sold)
sns.set_context("talk")
sns.set_style("dark")
plt.show()
X = df1.drop(["average_price"],axis="columns")
X.head()
y = df1["average_price"]
y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
reg = LinearRegression()
forest = RandomForestRegressor()
tree = DecisionTreeRegressor()
la = Lasso()

reg.fit(X_train,y_train)
forest.fit(X_train,y_train)
tree.fit(X_train,y_train)
la.fit(X_train,y_train)

pd1 = reg.predict(X_test)
pd2 = forest.predict(X_test)
pd3 = tree.predict(X_test)
pd4 = la.predict(X_test)

s1 = r2_score(y_test,pd1)
s2 = r2_score(y_test,pd2)
s3 = r2_score(y_test,pd3)
s4 = r2_score(y_test,pd4)

Scores = [s1,s2,s3,s4]
Models = ["LinearRegression","RandomForestRegressor","DecisionTreeRegressor","Lasso"]

plt.figure(figsize=[15,10])
plt.xticks(rotation=10,fontsize=20)
plt.yticks(fontsize=10)
plt.xlabel("Models",color='g',fontsize=10)
plt.ylabel("Scores",color='g',fontsize=10)
plt.title("Accuracy of Each Model",color="magenta")
sns.barplot(Models,Scores)
sns.set_context("talk")
sns.set_style("dark")

for i,v in enumerate(Scores):
    plt.text(i-.05,v+.01,format(Scores[i],'.2f'),fontsize=10)
    
plt.show()
    
data = {"Models" : ["LinearRegression","RandomForestRegressor","DecisionTreeRegressor","Lasso"],
       "Scores" : [s1,s2,s3,s4] }

df = pd.DataFrame(data)
df