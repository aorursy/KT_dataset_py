import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns #visualization

import matplotlib.pyplot as plt

%matplotlib inline
wines = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')
wines.head(5)
wines = wines.drop(columns=['Unnamed: 0','region_1','region_2','taster_twitter_handle','designation'])
wines.head(0)
wines.isnull().values.any()
sns.heatmap(wines.isnull(),yticklabels=False,cbar=False,cmap='viridis')
wines.price.fillna(wines.price.dropna().median(),inplace =True)

wines['taster_name'].fillna("No name", inplace =True)
sns.heatmap(wines.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_context("talk")

plt.figure(figsize=(30,5))

sns.boxplot(x=wines['price'],palette = 'colorblind')
sns.set_context("talk")

plt.figure(figsize=(30,5))

sns.boxplot(x=wines['points'],palette = 'colorblind')