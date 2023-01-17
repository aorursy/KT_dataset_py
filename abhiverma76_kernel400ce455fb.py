import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
data=pd.read_csv("../input/StudentsPerformance.csv")
data.head()
data.shape
data.shape
data.sample(5)
data.sample(frac=0.1)
data.info()
data.describe()
data.corr()
data['gender'].value_counts()
data['gender'].unique()
fig=plt.figure(figsize=(7,7))

ax=fig.add_subplot(111)
ax=sns.barplot(x=data['gender'].value_counts().index, y=data['gender'].value_counts().values,hue=['female','male'])

plt.xlabel('gender')

plt.ylabel('Frequency')

plt.title('Gender Bar Plot')

plt.show()
ax=sns.pointplot(x="reading score", y="writing score", hue="gender", data=data, markers=["o","x"],

                linestyle=["-","__"])

plt.xticks(rotation=90)

plt.show()
sns.countplot(x='gender',data=data)

plt.ylabel('Frequency')

plt.title('Gender Bar Plot')

plt.show()
sns.countplot(x='parental level of education',data=data)

plt.xticks(rotation=45)

plt.show()
sns.barplot(x='gender',y='reading score',data=data)

plt.show()
sns.barplot(x='gender',y='reading score',hue='race/ethnicity',data=data)

plt.show()
sns.distplot(data['writing score'],bins=10,kde=True)

plt.show()
sns.jointplot(x='math score',y='gender',data=data)

plot.show()
sns.pairplot(data)

plt.show()
sns.boxplot(x='gender',y='math score',data=data)

plt.show()
sns.boxplot(x='gender',y='writing score',data=data)

plt.show()
sns.boxplot(x='race/ethnicity',y='writing score',data=data)

plt.show()
data.corr()
sns.heatmap(data.corr())
g=sns.catplot(x='gender',y='writing score',hue='lunch',col='race/ethnicity',data=data,kind='bar')

plt.show()
g=sns.catplot(x='gender',y='writing score',hue='lunch',col='race/ethnicity',col_wrap=3,data=data,kind='bar',height=4,aspect=0.7)

plt.show()