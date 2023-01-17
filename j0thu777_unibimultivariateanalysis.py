import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline 
df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
df.head()
df.shape

df.tail()
df_setosa = df.loc[df['species']=='setosa' ]

df_virginica = df.loc[df['species']=='virginica']

df_versicolor = df.loc[df['species']=='versicolor']
plt.figure(figsize=(10, 6))

plt.plot(df_setosa['sepal_length'], np.zeros_like(df_setosa['sepal_length']), 'o')

plt.plot(df_virginica['sepal_length'], np.zeros_like(df_virginica['sepal_length']), 'o')

plt.plot(df_versicolor['sepal_length'], np.zeros_like(df_versicolor['sepal_length']), 'o')

sns.FacetGrid(df, hue='species', size=6).map(plt.scatter, 'petal_length', 'sepal_width').add_legend()

plt.show()
sns.pairplot(df, hue='species', size=3)