import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

print(os.listdir("../input"))
dataset = pd.read_csv('../input/master.csv')
#first 5 rows

dataset.head()
#last 5 rows

dataset.tail()
#random 5 rows

dataset.sample(5)
unique_country = dataset['country'].unique()

print(unique_country)

#unique country
#Info about the dataset

dataset.info()
#dataset Column

dataset.columns
#dataset shape

print('Data shape')

dataset.shape
#null value check

dataset.isnull().any()
dataset.isnull().values.any()
dataset.isnull().sum()
plt.figure(figsize=(10,25))

sns.countplot(y='country', data=dataset, alpha=0.7)

plt.title('Date by country')

plt.show()
plt.figure(figsize=(16,7))

#Plot the graph

sex = sns.countplot(x='sex', data=dataset)
plt.figure(figsize=(16,7))

cor = sns.heatmap(dataset.corr(), annot =True)
plt.figure(figsize=(16,7))

bar = sns.barplot(x='sex', y = 'suicides_no', hue='age', data= dataset)
plt.figure(figsize=(16,7))

bar = sns.barplot(x='sex', y = 'suicides_no', hue='generation', data= dataset)
dataset.groupby('age')['sex'].count()
sns.barplot(x=dataset.groupby('age')['sex'].count().index, y=dataset.groupby('age')['sex'].count().values)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(dataset.sex,hue=dataset.age)

plt.title('Sex and age')

plt.show()