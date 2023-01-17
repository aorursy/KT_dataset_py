import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



import statsmodels.api as sm



import seaborn as sns



# Input data file paths

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
df.info()
plt.scatter(df['longitude'], df['latitude'], s=1)

plt.axis('equal')

plt.axis('off')

plt.show()
fig, axes = plt.subplots(1,3, figsize=(12,3))

sns.distplot(df['price'], ax=axes[0])

df['price_log'] = df['price'].apply(np.log1p)

sns.distplot(df['price_log'], ax=axes[1])

sm.qqplot(df['price_log'], dist=scipy.stats.distributions.norm, fit=True, line='45', ax=axes[2])
