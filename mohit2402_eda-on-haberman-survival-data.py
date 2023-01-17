import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv("../input/habermancsv/haberman.csv")

data.head()
data.columns
data.count()
data.isnull().sum()
data['status'].value_counts()
data['status']=data['status'].apply(lambda x: 'survived' if x==1 else 'died')
s=sns.FacetGrid(data,hue='status',size=6)

s=s.map(sns.distplot,'age')

s.add_legend()

plt.show()
s=sns.FacetGrid(data,hue='status',size=6)

s.map(sns.distplot,'year')

s.add_legend()

plt.show()

s=sns.FacetGrid(data,hue='status',size=6)

s.map(sns.distplot,'nodes')

s.add_legend()

plt.show()
sns.set_style('whitegrid')

sns.pairplot(data,hue='status',height=5)

plt.show()
sns.boxplot(x='status',y='age',data=data)

plt.show()
sns.boxplot(x='status',y='year',data=data)

plt.show()
sns.boxplot(x='status',y='nodes',data=data)

plt.show()