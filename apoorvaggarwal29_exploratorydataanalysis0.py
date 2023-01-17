import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.info()
df.describe()
sns.heatmap(df.isnull(), cbar=False)

plt.show()
sns.heatmap(df.corr(),annot=True)

plt.show()
df['avg score']=(df['math score']+df['reading score']+df['writing score'])/3
df.head()
scores=['math score','reading score','writing score','avg score']
print('uniqe values:''\n''\n') 

print('test prep course:' )

print(df['test preparation course'].value_counts())

print('\n')

print('lunch:')

print(df['lunch'].value_counts())

print('\n')

print('parent education:')

print(df['parental level of education'].value_counts())



plt.rcParams['figure.figsize'] = (18, 6)

plt.subplot(1, 3, 1)

sns.distplot(df['math score'])



plt.subplot(1, 3, 2)

sns.distplot(df['reading score'])



plt.subplot(1, 3, 3)

sns.distplot(df['writing score'])



plt.suptitle('Checking for Skewness', fontsize = 18)

plt.show()
sns.countplot('gender',data=df)

plt.show()
plt.figure(figsize=(10,7))

sns.countplot('parental level of education',data=df)

plt.show()
fig, ax = plt.subplots(figsize=(15,7))

df.groupby('parental level of education')[scores].mean().plot.bar(ax=ax)

plt.ylabel('scores')

plt.show()
fig, ax = plt.subplots(figsize=(15,7))

df.groupby('gender')[scores].mean().plot.bar(ax=ax)

plt.ylabel('score')

plt.show()
fig, ax = plt.subplots(figsize=(15,7))

df.groupby('race/ethnicity')[scores].mean().plot.bar(ax=ax)

plt.ylabel('scores')

plt.show()
fig, ax = plt.subplots(figsize=(15,7))

df.groupby('test preparation course')[scores].mean().plot.bar(ax=ax)

plt.ylabel('scores')

plt.show()
fig, ax = plt.subplots(figsize=(15,5))

df.groupby('lunch')[scores].mean().plot.bar(ax=ax)

plt.ylabel('scores')

plt.show()