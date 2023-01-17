import pandas as pd

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/fertilizers-by-product-fao/FertilizersProduct.csv', encoding='latin-1')
df.head()
plt.figure(figsize=(10,6))

plt.tight_layout()

sns.set_style('whitegrid')

sns.countplot(x='Year',data=df)
IndiaData = df[df['Area']=='India']
plt.figure(figsize=(10,6))

IndiaData['Year'].hist(bins=30)
plt.figure(figsize=(10,6))

sns.countplot(x='Year',data = IndiaData,hue='Element')
plt.figure(figsize=(15,8))

chart = sns.countplot(x = 'Item', data=IndiaData)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')

chart.plot()
plt.figure(figsize=(15,8))

chart = sns.countplot(x = 'Item', data=IndiaData[IndiaData['Element']=='Agricultural Use'])

chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')

chart.plot()
plt.figure(figsize=(10,6))

dat = IndiaData[IndiaData['Element']=='Production']

chart = sns.countplot(x = 'Item', data=dat)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')

chart.plot()
plt.figure(figsize=(10,6))

export_data = IndiaData[IndiaData['Element']=='Export Value']

sns.lineplot(x = 'Year', y='Value', data=export_data)
chart = sns.barplot(x='Item',y='Value',data=export_data)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')

chart.plot()