import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb
data = pd.read_csv('../input/Mall_Customers.csv')

data.info()
data.shape
data.head()
data.corr()
f, ax = plt.subplots(figsize=(13,6))

sb.heatmap(data.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax)

plt.show()
data.plot(kind='scatter', x='Age', y='Spending Score (1-100)', color='red', alpha=0.5, figsize=(7,7))

plt.xlabel('Age')

plt.ylabel('Spending score')

plt.show()
data.Age.plot(kind='hist', figsize=(15,8), bins=50)

plt.xlabel('Age')

plt.show()