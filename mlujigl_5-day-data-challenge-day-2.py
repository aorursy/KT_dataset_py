import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
ds = pd.read_csv('../input/cereal.csv',sep=',',skiprows=range(1,2))

ds.head(3)
ds.describe()
ds.isnull().any().any(),ds.shape
% matplotlib inline

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



ds.hist(column = 'calories',figsize=(10,5))

cal_counts = ds['calories'].value_counts()

cal_counts[-5:]

cal_counts.plot(kind='bar',figsize=(10,5),color='red',legend='calories',label='calories')

plt.show()
ds.columns
ds.index
ds.index
ds['calories'].mean()
ds[ds['calories'] <= 70]
ds.corr(method='pearson')
ds[['name','calories']].head()
most_cal = ds['calories'] >= 130

most_sug = ds['sugars'] >= 5

rank_count = ds[['calories','sugars']].groupby('sugars').count()

rank_count