# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.shape     #checking how many rows and columns there are
data.columns         #We will check the columns first
data.head()      #checking the first 5 rows to get an insight
data.tail()    #checking the last 5 rows to get an insight.
data.dtypes       #checking data types of columns
#frequency of chest pain types

data["cp"].value_counts(dropna=False)
data["age"].describe()
#checking Q1, Q2, Q3 values manually

data1=data["age"].sort_values()   #data1 is sorted but index numbers did not change.

data2=[i for i in data1] #data2 has the same values as data1, but index numbers are reassigned, they goes from 0 to forward.

print("Q1:",data2[75])

print("median:",data2[151])

print("Q3:",data2[227])
data[["age"]].boxplot()   #There is no outlier as we can see below.

plt.show()
data.boxplot(column="age")

plt.show()
data.head()
#tidy data (melting)

data3=data.head()

print(data3)



melted=pd.melt(data3,id_vars="age",value_vars=["trestbps","oldpeak"])

melted
#pivoting melted data

melted.pivot(index="age",columns="variable",values="value")
#concatenating data

data4=data.loc[0:4,["trestbps","chol"]]

data5=data.loc[0:4,["fbs","restecg"]]

concat1=pd.concat([data4,data5],axis=1,sort=False)

print(concat1)

print("")

concat2=pd.concat([data4,data5],axis=0,sort=False,ignore_index=True)

print(concat2)
#data types

print(data.dtypes)

data["sex"]=data["sex"].astype("int32")      #changing data type of "sex" column

data["trestbps"]=data["trestbps"].astype("category")   #changing data type of "trestbps" column

print(data.dtypes)      #new data types
concat2["trestbps"].value_counts(dropna=False)    #determining NaN values in trestbps column of concat2 dataframe
concat2["trestbps"].dropna(inplace=True)     #dropping NaN values

concat2["trestbps"].value_counts(dropna=False)   #check if we could drop NaN values
assert concat2["trestbps"].notnull().all()    #check if we could drop NaN values
concat2["trestbps"].fillna("empty",inplace=True)    #filling NaN values

concat2["trestbps"].value_counts(dropna=False)      #check the trestbps column