import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/StudentsPerformance.csv")
df.head()
# To check missing values and other statistics of data
df.describe()
# Check missing values
df.isnull().sum()
print(df['gender'].value_counts())
df['gender'].value_counts().plot.bar()
df['race/ethnicity'].value_counts().plot.bar()
df['parental level of education'].value_counts().plot.bar()
#df['math score'].plot.hist()
sns.distplot(df['math score'], bins=10, kde=False)
g = sns.FacetGrid(df, col="gender")
g.map(sns.kdeplot, "math score")
g.map(sns.kdeplot,"reading score")
sns.pairplot(df[['math score', 'reading score', 'writing score']])
#Understanding number of males and females in top 10 of math score
df[['gender','math score']].sort_values(by='math score', ascending=False).iloc[:10].gender.value_counts()
#Understanding number of males and females in top 10 of reading score 
df[['gender','reading score']].sort_values(by='reading score', ascending=False).iloc[:10].gender.value_counts()
#Understanding number of males and females in top 10 of writing score 
df[['gender','writing score']].sort_values(by='writing score', ascending=False).iloc[:10].gender.value_counts()
df[['gender','math score']].sort_values(by='math score', ascending=False).iloc[:50].gender.value_counts().plot.bar()
df[['gender','reading score']].sort_values(by='reading score', ascending=False).iloc[:50].gender.value_counts().plot.bar()
df[['gender','writing score']].sort_values(by='writing score', ascending=False).iloc[:50].gender.value_counts().plot.bar()
sns.kdeplot(df[df['math score'] >40]['math score'])
sns.countplot(df[df['math score'] >90]['math score'])
#Intial EDA
df.groupby('parental level of education')['math score'].agg([len, min, max])
#df.groupby('parental level of education')['reading score'].agg([len, min, max])
#df.groupby('parental level of education')['writing score'].agg([len, min, max])
group_math = df.sort_values(by='math score', ascending=False).iloc[:50]
group_math.groupby('parental level of education')['math score'].agg([len, min, max])
sum_ = df['math score'] + df['reading score'] + df['writing score']
df['avg'] = sum_/3
df.head()
df.agg([len, min, max])
#To understand number of students with average more than 80 and maximum they've scored
df[df['avg']>80].groupby('parental level of education')['avg'].agg([len, min, max]).plot.bar()
(df[df['avg']>75].groupby('parental level of education')['avg'].agg([len])/df.groupby('parental level of education')['avg'].agg([len])).plot.bar()
(df[df['avg']>80].groupby('parental level of education')['avg'].agg([len])/df.groupby('parental level of education')['avg'].agg([len]))
g = sns.FacetGrid(df, col="parental level of education", col_wrap=1)
g.map(sns.kdeplot, "math score")
#Initial EDA
df.groupby('test preparation course').size()
df.groupby('test preparation course')['parental level of education'].value_counts()
df.groupby('test preparation course')['parental level of education'].value_counts()/df.groupby('parental level of education').size()
df.groupby('parental level of education').size()
df[df['avg']>85].groupby('test preparation course')['test preparation course'].value_counts()
temp = df.sort_values(by='math score', ascending=False).iloc[:20]
temp.groupby('test preparation course')['test preparation course'].value_counts()
temp = df.sort_values(by='reading score', ascending=False).iloc[:20]
temp.groupby('test preparation course')['test preparation course'].value_counts()
temp = df.sort_values(by='writing score', ascending=False).iloc[:20]
temp.groupby('test preparation course')['test preparation course'].value_counts()
