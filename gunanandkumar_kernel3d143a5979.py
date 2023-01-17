import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import math

%matplotlib inline

matplotlib.rcParams['figure.figsize']=(20,10)
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
train_df.head(15)
test_df.head()
train_df.info()
train_df.describe()
train_df.shape
train_df.isnull().sum()
test_df.describe()
test_df.shape
sns.pairplot(hue = 'Sex',data = train_df)
sns.countplot(x='Sex',data = train_df)
sns.countplot(x = 'SmokingStatus',data = train_df)
sns.countplot(x='SmokingStatus',hue = 'Sex',data = train_df)
plt.scatter(x='Age',y='FVC',data = train_df)
train_df.Patient.value_counts()
train_df.Patient.dtype
labels = train_df['SmokingStatus'].value_counts().index

values = train_df['SmokingStatus'].value_counts().values
plt.pie(values,labels = labels,autopct = '%0.1f%%')

plt.show()
plt.pie(values,labels = labels,radius = 1,autopct = '%0.1f%%')

plt.pie([1],colors = ['w'],radius = 0.5)
labels1 = train_df['Sex'].value_counts().index

values1 = train_df['Sex'].value_counts().values
plt.pie(values1,labels = labels1,autopct = '%0.1f%%')

plt.show()
plt.pie(values1,labels = labels1,radius = 1,autopct = '%0.1f%%')

plt.pie([1],colors = ['w'],radius=0.5)

plt.show()
set1 = dict(Male = 'red',Female = 'green')
a = sns.FacetGrid(train_df,hue = 'Sex',aspect = 4,palette = set1)

a.map(sns.kdeplot,'FVC',shade = True)

a.set(xlim=(0,train_df['FVC'].max()))

a.add_legend()
train_df['SmokingStatus'].unique()