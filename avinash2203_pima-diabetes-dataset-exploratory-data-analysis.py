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

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style("whitegrid")
data = pd.read_csv('../input/diabetes.csv') #importing dataset

data.info()
data.describe()
corr= data.corr()

sns.heatmap(corr,annot=True)
sns.pairplot(data,hue='Outcome')
sns.distplot(data['Age'],kde=True,rug=True)
sns.distplot(data['Pregnancies'],kde=True,rug=True)
sns.distplot(data['Insulin'],kde=True,rug=True)
sns.jointplot(data['Age'],data['Pregnancies'])
sns.jointplot(data['Glucose'],data['Outcome'],kind='reg')
sns.countplot(x='Pregnancies',hue='Outcome',data=data)


g = sns.FacetGrid(data,col='Outcome')

g.map(plt.hist,'Age')
insulin_0 = data[(data['Outcome']==0) & (data['Insulin']!=0)]['Insulin'].mean()

insulin_1 = data[(data['Outcome']==1) & (data['Insulin']!=0)]['Insulin'].mean()

print(insulin_0)

print(insulin_1)
glucose_0 = data[(data['Outcome']==0) & (data['Glucose']!=0)]['Glucose'].mean()

glucose_1 = data[(data['Outcome']==1) & (data['Glucose']!=0)]['Glucose'].mean()

print(glucose_0)

print(glucose_1)
skin_0 = data[(data['Outcome']==0) & (data['SkinThickness']!=0)]['SkinThickness'].mean()

skin_1 = data[(data['Outcome']==1) & (data['SkinThickness']!=0)]['SkinThickness'].mean()

print(skin_0)

print(skin_1)
from scipy.stats import iqr
print(iqr(data[(data['Outcome']==0) & (data['Glucose']!=0)]['Glucose'],rng=(25,75)))



print(iqr(data[(data['Outcome']==1) & (data['Glucose']!=0)]['Glucose'],rng=(25,75)))
print(iqr(data[(data['Outcome']==0) & (data['Insulin']!=0)]['Insulin'],rng=(25,75)))



print(iqr(data[(data['Outcome']==1) & (data['Insulin']!=0)]['Insulin'],rng=(25,75)))
plt.figure(figsize=(20,10))

sns.boxplot(data=data)
plt.figure(figsize=(20,10))

sns.boxplot(x='Outcome',y='Insulin',data=data)
data_new = data[data['Insulin']!=0]

plt.figure(figsize=(20,10))

sns.boxplot(x='Outcome',y='Insulin',data=data_new)
plt.figure(figsize=(20,10))

sns.boxplot(x='Outcome',y='Glucose',data=data)
data_new = data[data['Glucose']!=0]

plt.figure(figsize=(20,10))

sns.boxplot(x='Outcome',y='Glucose',data=data_new)
data_new = pd.DataFrame()

data_new['Glucose'] = data['Glucose'].replace(0,data['Glucose'].median())

data_new['Insulin'] = data['Insulin'].replace(0,data['Insulin'].median())

data_new['BMI'] = data['BMI'].replace(0,data['BMI'].median())

data_new['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].median())

data_new['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].median())

data_new['Outcome'] = data['Outcome']
data_new.head()
plt.figure(figsize=(20,10))

sns.boxplot(x='Outcome',y='Glucose',data=data_new)
plt.figure(figsize=(20,10))

sns.boxplot(x='Outcome',y='Insulin',data=data_new)
plt.figure(figsize=(20,10))

sns.boxplot(x='Outcome',y='BMI',data=data_new)
print(iqr(data_new[(data_new['Outcome']==0)]['Glucose'],rng=(25,75)))



print(iqr(data_new[(data_new['Outcome']==1)]['Glucose'],rng=(25,75)))
print(iqr(data_new[(data_new['Outcome']==0)]['Insulin'],rng=(25,75)))



print(iqr(data_new[(data_new['Outcome']==1)]['Insulin'],rng=(25,75)))
plt.figure(figsize=(20, 10))

sns.violinplot(x='Outcome',y='Glucose',data=data_new,inner='quartile')