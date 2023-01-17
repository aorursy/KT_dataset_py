import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data = pd.read_csv('../input/WATCHLIST.csv', encoding='latin-1')
data.head()
plt.figure(figsize=(18, 8))

sns.distplot(data['IMDb Rating'],kde = True)

plt.show()
plt.figure(figsize=(18, 8))

ax = sns.countplot(x="Title Type", data=data);
plt.figure(figsize=(18, 8))

sns.distplot(data[['Your Rating']], hist=False, kde_kws={"label": "My Rating"})

sns.distplot(data[['IMDb Rating']], hist=False, kde_kws={"label": "IMDb Rating"})



plt.show();