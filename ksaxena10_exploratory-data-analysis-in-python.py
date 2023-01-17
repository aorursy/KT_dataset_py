# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("./../input/craigslist-carstrucks-data/vehicles.csv")
df.head()
df.info()
df['price'].describe() # For descriptive analysis of price
df['manufacturer'].value_counts().head(30).plot(kind='barh', figsize=(4,10))
sns.catplot(x="cylinders", hue="fuel", row="type", data=df, kind="count", height=3, aspect=4)
plt.figure(figsize=(30, 20))

sns.countplot(df['type'])
plt.figure(figsize=(30, 20))

sns.countplot(df['type'], hue=df['drive'])
plt.figure(figsize=(30, 20))

sns.countplot(df['type'], hue=df['transmission'])
plt.figure(figsize=(30, 20))

sns.countplot(df['type'], hue=df['fuel'])
plt.figure(figsize=(30, 20))

sns.countplot(df['type'], hue=df['cylinders'])
plt.figure(figsize=(30, 20))

sns.countplot(df['type'], hue=df['size'])
plt.figure(figsize=(30, 20))

sns.barplot(df['type'], df['price'], hue= df['transmission'])
plt.figure(figsize=(30, 20))

sns.pointplot(df['type'], df['price'], hue=df['drive'])
plt.figure(figsize=(10, 10))

sns.stripplot(df['fuel'], df['price'], jitter= True)
plt.figure(figsize=(10, 10))

sns.jointplot(df['odometer'], df['price'])