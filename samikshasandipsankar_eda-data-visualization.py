import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv(r'../input/exploratory-data-visualization/StudentsPerformance.csv')
df.head()
corelation = df.corr()
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns ,annot=True)
sns.pairplot(df)
sns.countplot(x='race/ethnicity',data=df)
sns.countplot(x='race/ethnicity',hue='parental level of education',data=df)
sns.lineplot(x='reading score',y='writing score',data=df)
sns.relplot(x='reading score',y='writing score',hue='gender',data=df)
df.plot.scatter(x='gender',y='math score')
sns.distplot(df['writing score'],bins=5)
sizes=df['gender'].value_counts()

fig1,ax1=plt.subplots()

ax1.pie(sizes,labels=['male','female'],autopct='%1.2f%%',shadow=True)

plt.show()