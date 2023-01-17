import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import os
os.listdir("../input")
data = pd.read_csv("../input/train.csv")
data.head()
print(data.dtypes)
print("Total count:", len(data))
print()
print(round((data.Survived.value_counts()/len(data)) * 100,2))
def get_uniquevals(df):
    print("-"*40)
    for col in df.columns:
        if len(df[col].unique()) <= 10:
            print("{} - Unqiue Values:".format(df[col].name))
            print(df[col].unique())
            print()
            print("{} - # of occurences of each values:".format(df[col].name))
            print(df[col].value_counts())
        else:
            print("{} has {} unqiue values:".format(df[col].name,len(df[col].unique())))
        print("-"*40)
get_uniquevals(data)
def getnullcounts(df):
    print("-"*20)
    non_nullcols = []
    for col in df.columns:
        if df[col].isna().sum() > 0:
            print("{} : {}".format(df[col].name, df[col].isna().sum()))
        else:
            non_nullcols.append(df[col].name)
    print("-"*20)
    print('Non-null features:\n',', '.join(non_nullcols))
    print("-"*20)
getnullcounts(data)
def feature_elimination(df):
    print('Features to be considered for elimiation:')
    for col in df.columns:
        if len(df[col].unique()) == (len(df)) and df[col].dtype != 'object':
            print(df[col].name)
        if len(df[col].unique()) > (len(df)*0.50) and df[col].dtype == 'object':
            print(df[col].name)
feature_elimination(data)
f, ax = plt.subplots(figsize=(11,5))
sns.boxplot(x='Survived', y="Age",  data=data);
f, ax = plt.subplots(figsize=(11,5))
sns.boxplot(x="Sex", y="Age", hue="Survived", data=data);
f, ax = plt.subplots(figsize=(7,3))
sns.barplot(x='Sex', y="Survived",  data=data);
sns.barplot(x="Pclass", y="Survived", data=data);
sns.barplot(x="Pclass", y="Survived",hue="Sex", data=data);
sns.barplot(x="SibSp", y="Survived", data=data);
sns.barplot(x="Parch", y="Survived", data=data);
data["family"] = data["SibSp"] + data["Parch"]
data["occumpanied"] = data["family"].apply(lambda x: 0 if x == 0 else 1)
sns.barplot(x="Survived", y="occumpanied", data=data);
sns.distplot(data['Age'].dropna());
survived = data.loc[data['Survived']==1,"Age"].dropna()
sns.distplot(survived)
plt.title("Survived");
not_survived = data.loc[data['Survived']==0,"Age"].dropna()
sns.distplot(not_survived)
plt.title("Not Survived");
sns.pairplot(data.dropna());