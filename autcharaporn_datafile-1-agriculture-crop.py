import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/agricuture-crops-production-in-india/datafile (1).csv')

df.head()
df.info()
df.dtypes
print("Are There Missing Data? :",df.isnull().any().any())

print(df.isnull().sum())
df.corr()
f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(df.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)

plt.show()