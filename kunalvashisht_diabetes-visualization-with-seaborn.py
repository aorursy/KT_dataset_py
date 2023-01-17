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
from matplotlib import pyplot as plt
df = pd.read_csv("../input/diabetes.csv")
df.head()
df.tail()
df.sample(6)
df.shape
df.dtypes
df.columns
df.describe()
print("pregnancies count are",df['Pregnancies'].count())
df.isnull().sum() #no missing values in the dataset...
sns.barplot(x='Outcome',y='BloodPressure',data=df,hue="Outcome")
plt.figure(figsize=(12,12))
plt.subplot(3,3,1)
sns.barplot(x='Outcome',y='Glucose',data=df,hue="Outcome")
plt.subplot(3,3,2)
sns.barplot(x='Outcome',y='BloodPressure',data=df,hue="Outcome")
plt.subplot(3,3,3)
sns.barplot(x='Outcome',y='SkinThickness',data=df,hue="Outcome")
plt.subplot(3,3,4)
sns.barplot(x='Outcome',y='BMI',data=df,hue="Outcome")
plt.subplot(3,3,5)
sns.barplot(x='Outcome',y='DiabetesPedigreeFunction',data=df,hue="Outcome")
plt.subplot(3,3,6)
sns.barplot(x='Outcome',y='Age',data=df,hue="Outcome")
sns.factorplot(x='Pregnancies',y='Insulin',data=df,hue='Outcome')
sns.swarmplot(x='Outcome',y='BloodPressure',data=df,hue='Outcome')
sns.lmplot(x='Pregnancies',y='Glucose',data=df,hue="Outcome")
aa=sns.pairplot(df,hue='Outcome')
plt.figure(figsize=(12,12))
plt.subplot(3,3,1)
sns.distplot(df.Pregnancies)
plt.subplot(3,3,2)
sns.distplot(df.Glucose)
plt.subplot(3,3,3)
sns.distplot(df.BloodPressure)
plt.subplot(3,3,4)
sns.distplot(df.SkinThickness)
plt.subplot(3,3,5)
sns.distplot(df.BMI)
plt.subplot(3,3,6)
sns.distplot(df.DiabetesPedigreeFunction)
sns.boxplot(x="Pregnancies",y="Age",data=df,hue="Outcome")
sns.countplot(x="Pregnancies",data=df)
sns.countplot(x="Outcome",data=df)
