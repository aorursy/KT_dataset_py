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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import torch

import torch.nn as nn
df = pd.read_csv(os.path.join(dirname,filename))
df.head()
df.shape
df.isnull().sum()
df["Drug"].value_counts() # Drug B and Drug C are under represented
df.describe()
sns.set(font_scale=1)

plt.title("Distribution of Age")

sns.distplot(df["Age"])
plt.figure(figsize=(12,8))

plt.title("Boxplot for ages")

sns.boxplot(x="Drug",y="Age",data=df)
df.groupby("Drug").mean()["Age"].sort_values(ascending=False)
df["Sex"].value_counts()
plt.figure(figsize=(12,8))

plt.title("Drug wise Gender Distribution")

sns.countplot(x="Sex",hue="Drug",data=df)
plt.figure(figsize=(12,10))

plt.title("Boxplot of Age and Distribution")

sns.boxplot(x="Sex",y="Age",data=df)
df.head()
def one_hot_encode(dataframe, col):

    dummy = pd.get_dummies(dataframe[col],drop_first=True)

    dummy.columns = [str(i)+"_dumy_"+col for i in range(len(dummy.columns))]

    dataframe = dataframe.join(dummy)

    dataframe.drop([col],axis=1,inplace=True)

    return dataframe
df = one_hot_encode(df,"Sex")
df.head()
plt.figure(figsize=(12,10))

plt.title("Boxplot of Blood Pressure")

sns.boxplot(x="BP",y="Age",data=df)

#One hot Encode the BP variable

df = one_hot_encode(df,"BP")
df.head()
sns.distplot(df["Na_to_K"])

#There seems to be some outliers present
sns.boxplot(df["Na_to_K"])
plt.figure(figsize=(12,9))

sns.boxplot(x="Drug",y="Na_to_K",data=df)
df[df.Na_to_K>32]
plt.figure(figsize=(12,10))

sns.scatterplot(x="Age",y="Na_to_K",data=df,hue="Drug",s=500,alpha=0.7)
df["Cholesterol"].value_counts()
df = one_hot_encode(df,"Cholesterol")
df.head()
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df["Drug"] = lb.fit_transform(df["Drug"])
df.head()
lb.classes_
df.columns
numerical_feats = df[["Age",'Na_to_K']]
numerical_feats
X = df.drop(["Drug"],axis=1).values

y = df["Drug"].values
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=15,random_state=42,stratify=y)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_X_train = scaler.fit_transform(X_train)

scaled_X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, p=2 )

knn.fit(scaled_X_train,y_train)
pred_k = knn.predict(scaled_X_test)
from sklearn import metrics

def evaluate_result(true,pred):

    print(metrics.classification_report(true,pred))

    print(metrics.accuracy_score(true,pred))
evaluate_result(y_test,pred_k)
sc = MinMaxScaler()

scaled_X = sc.fit_transform(X)
predictions = knn.predict(scaled_X)
df=df.join(pd.DataFrame(predictions,columns=["predictions"]))
df[["Drug","predictions"]]
evaluate_result(df.Drug,df.predictions)
plt.figure(figsize=(15,8))

sns.scatterplot(x='Age',y="Na_to_K",hue="Drug",data=df,s=500,alpha=0.5)
plt.figure(figsize=(15,8))

sns.scatterplot(x='Age',y="Na_to_K",hue="predictions",data=df,s=500,alpha=0.5)