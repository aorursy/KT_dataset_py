import numpy as np

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns #Graph library that use matplot in background

import matplotlib.pyplot as plt #to plot some parameters in seaborn



#Importing the data

df = pd.read_csv("/kaggle/input/german-credit-data-risk/german_credit_data.csv",index_col=0)
df.head()
df.nunique()
df["Risk"].unique()
df["Housing"].unique()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()



df["Risk"] = LE.fit_transform(df["Risk"])

df["Housing"] = LE.fit_transform(df["Housing"])

df.head()
df.info()
df.isnull().sum()
df["Checking account"].unique()
df["Checking account"] = df["Checking account"].map({ 'little' : 1, 'moderate': 2 , 'rich': 3})
df["Checking account"]  = df["Checking account"].fillna(0)
df["Checking account"].unique()
df["Purpose"].unique()


sns.catplot(x="Purpose", hue="Risk", kind="count", data=df, height=8.27, aspect=11.7/8.27)
df["Purpose"] = df["Purpose"].map({ 'radio/TV' : 1, 'education' : 2 , 'furniture/equipment' : 1 , 'car' : 0 , 'business' :3,

       'domestic appliances' : 1 , 'repairs' : 1 , 'vacation/others' : 4})
df["Saving accounts"].unique()
df["Saving accounts"] = df["Saving accounts"].map({ 'little' : 1, 'moderate': 2 , 'rich': 4 , 'quite rich' : 3})
df["Saving accounts"] = df["Saving accounts"].fillna(0)
df["Saving accounts"].unique()
df.info()
df["male"] = pd.get_dummies(df["Sex"], drop_first=True)
df = df.drop("Sex" , axis = 1 )

df.info()
sns.catplot(x="Housing", hue="Risk", kind="count", data=df);
sns.barplot(x="Saving accounts", y="Risk", hue = "Purpose", data=df)
sns.catplot(x="male", hue="Risk", kind="count", data=df, height=8.27, aspect=11.7/8.27)
X = df.drop("Risk", axis = 1)

X.shape
y = df[["Risk"]]

y.shape
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

knn.score(X_train, y_train)
knn.score(X_test, y_test)