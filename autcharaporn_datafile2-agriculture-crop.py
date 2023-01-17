import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/agricuture-crops-production-in-india/datafile (2).csv')

df.head()
df.shape
df.info()
df.dtypes
print("Are There Missing Data? :",df.isnull().any().any())

print(df.isnull().sum())
df.corr()
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(df.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)

plt.show()