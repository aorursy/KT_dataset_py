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
#importing all the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#reading the csv file and creating the dataframe
df = pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")
df
df.shape

df.columns.tolist()
#Fixing the empty values represented by "?" in dataset
freq = df["workclass"].value_counts().idxmax()
df.loc[df["workclass"] == "?", "workclass"] =freq

freq = df["occupation"].value_counts().idxmax()
df.loc[df["occupation"] == "?", "occupation"] =freq

freq = df["native-country"].value_counts().idxmax()
df.loc[df["native-country"] == "?", "native-country"] =freq
df.head(10)
#Summary of categorical data
df.describe(include=["O"])
#summary of numerical data
df.describe()
df.head(10)
df.age.hist().set_title("Histogram of age")
plt.show()
df.fnlwgt.hist().set_title("Histogram of fnlwgt")
plt.show()
df["educational-num"].hist().set_title("Histogram of educational-num")
plt.show()
df["capital-gain"].hist().set_title("Histogram of capital-gain")
plt.show()
df["capital-loss"].hist().set_title("Histogram of capital-loss")
plt.show()
df["hours-per-week"].hist().set_title("Histogram of hours-per-week")
plt.show()
plt.figure(figsize=(10,5))
sns.countplot(x='workclass',data=df)
plt.title("Countplot of workclass")
plt.show()
plt.figure(figsize=(18,5))
sns.countplot(x='education',data=df)
plt.title("Countplot of education")
plt.show()
plt.figure(figsize=(13,5))
sns.countplot(x='marital-status',data=df)
plt.title("Countplot of marital-status")
plt.show()
plt.figure(figsize=(22,5))
sns.countplot(x='occupation',data=df)
plt.title("Countplot of occupation")
plt.show()
plt.figure(figsize=(10,5))
sns.countplot(x='relationship',data=df)
plt.title("Countplot of relationship")
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(x='race',data=df)
plt.title("Countplot of race")
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(x='gender',data=df)
plt.title("Countplot of gender")
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(y='native-country',data=df)
plt.title("Countplot of native-country")
plt.show()
plt.figure(figsize=(5,5))
sns.countplot(x='income',data=df)
plt.title("Countplot of income")
plt.show()
sns.catplot(x="income", y="age", data=df, kind ="box", height = 8, aspect=.5)
plt.title("Boxplot of income vs age")
plt.show()
sns.catplot(x="income", y="educational-num", data=df, kind ="box")
plt.title("Boxplot of income vs educational-num")
plt.show()
sns.catplot(x="income", y="capital-loss", data=df, kind ="box")
plt.title("Boxplot of income vs capital-loss")
plt.show()
sns.catplot(x="income", y="capital-gain", data=df, kind ="box")
plt.title("Boxplot of income vs capital-gain")
plt.show()
sns.catplot(x="income", y="hours-per-week", data=df, kind ="box", height=8,aspect=.75)
plt.title("Boxplot of income vs hours-per-week")
plt.show()
plt.figure(figsize=(10,5))
sns.boxplot(y="fnlwgt",x='income',data=df)
plt.title("Boxplot of income vs fnlwgt")
plt.show()
plt.figure(figsize=(10,5))
sns.countplot(x="workclass",hue='income',data=df)
plt.title("Countplot of workclass (hue - income)")
plt.show()
plt.figure(figsize=(10,5))
sns.countplot(y="education",hue='income',data=df)
plt.title("Countplot of education (hue - income)")
plt.show()
plt.figure(figsize=(13,5))
sns.countplot(x="marital-status",hue='income',data=df)
plt.title("Countplot of marital-status (hue - income)")
plt.show()
plt.figure(figsize=(13,6))
sns.countplot(y="occupation",hue='income',data=df)
plt.title("Countplot of occupation (hue - income)")
plt.show()
plt.figure(figsize=(13,6))
sns.countplot(y="relationship",hue='income',data=df)
plt.title("Countplot of relationship (hue - income)")
plt.show()
plt.figure(figsize=(8,4))
sns.countplot(y="race",hue='income',data=df)
plt.title("Countplot of race (hue - income)")
plt.show()
plt.figure(figsize=(5,4))
sns.countplot(y="gender",hue='income',data=df)
plt.title("Countplot of gender (hue - income)")
plt.show()
plt.figure(figsize=(10,8))
sns.countplot(y="native-country",hue='income',data=df)
plt.title("Countplot of native-country (hue - income)")
plt.show()
sns.pairplot(df)
plt.title("Pairplot of all numerical attributes")
plt.show()
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), annot = True)
plt.title("Heatmap")
plt.show()
plt.figure(figsize=(12,7))
sns.boxplot(x='income',y ='hours-per-week', hue='gender',data=df)
plt.title("Boxplot of income vs hours-per-week")
plt.show()
plt.figure(figsize=(12,7))
sns.boxplot(x='income',y ='age', hue='gender',data=df)
plt.title("Boxplot of income vs gender")
plt.show()