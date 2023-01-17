import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as plt
df=pd.read_csv("../input/titanicdataset-traincsv/train.csv")
pd.set_option("max_columns",40)
df.head()
df.shape #it has 891 rows and 12 columns
df.ndim #it is 2d data
df.describe()
df.info() #it shows age has some missing values
df["Age"].isnull().sum()  #Age has 177 missing values
df["Age"].nunique()
df["Age"].mode()
df["Age"]=df["Age"].fillna(24)
df["Age"].isnull().sum()
df.boxplot("Age")
u=df["Age"].mean()
std=df["Age"].std()
otlr=[]

for i in df["Age"]:

    z=(i-u)/std

    if z>3:

        otlr.append(i)
print(otlr) # These are the outliers
q1=df["Age"].quantile(.25)

q1
q3=df["Age"].quantile(.75)

q3
iqr=q3-q1
iqr
outlier=q3+(1.5*iqr)
outlier
df.groupby(["Sex","Pclass"])["Age"].count()
sns.barplot(x="Sex",y="Age",hue="Pclass",data=df)
df["Age"].hist()
sns.boxplot(x="Sex",y="Age",data=df)
sns.heatmap(df.corr(),annot=True)
df["SibSp"].unique()
df["Parch"].unique()
sns.barplot(x="Survived",y="Age",hue="SibSp",data=df)
sns.barplot(x="Survived",y="Age",hue="Parch",data=df)
df.groupby(["Survived","Sex"])["Age"].count()
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df)