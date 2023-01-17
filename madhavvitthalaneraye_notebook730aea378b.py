import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df=pd.read_csv(r'../input/real-estate-dataset/data.csv')

df.head()
corelation = df.corr()
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns ,annot=True)
sns.pairplot(df)
sns.countplot(x='INDUS',data=df)
sns.countplot(x='INDUS',hue='DIS',data=df)
sns.lineplot(x='RM',y='AGE',data=df)
sns.relplot(x='CRIM',y='ZN',hue='INDUS',data=df)
df.plot.scatter(x='ZN',y='INDUS')
sns.distplot(df['INDUS'],bins=5)
sizes=df['PTRATIO'].value_counts()

fig1,ax1=plt.subplots()

ax1.pie(sizes,labels=['17.8','18.7'],autopct='%1.2f%%',shadow=True)

plt.show()