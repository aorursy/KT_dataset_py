import pandas as pd

import numpy as np

import scipy as sp



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df_beer = pd.read_csv('../input/beers.csv')

df_beer.drop(['Unnamed: 0'], axis=1, inplace=True)
# We have some dublicated rows (example: brewery_id = 154)

df_beer.drop_duplicates(['brewery_id','style','abv'], inplace=True)



df_beer.drop(df_beer[df_beer['abv'].isnull() == True].index, axis=0, inplace=True)
df_beer.head()
# Finde mean value of abv for earch brewery and style

df_tmp = df_beer[['brewery_id','abv','style']].groupby(['brewery_id','style'], as_index=False).mean()
# Find absolut spread of abv for earch brewery



df_max = df_tmp[['brewery_id','abv']].groupby(['brewery_id'], as_index=False).max()

df_min = df_tmp[['brewery_id','abv']].groupby(['brewery_id'], as_index=False).min()



df_tmp_2 = df_beer[['abv','brewery_id']].groupby(['brewery_id'], as_index=False).mean()

df_tmp_2['abs_width'] = df_max['abv'].values - df_min['abv'].values
df_tmp_2['width_type'] = ''



df_tmp_2.ix[df_tmp_2['abs_width'] < 0.03,'width_type'] = 'single'

df_tmp_2.ix[df_tmp_2['width_type'] == '','width_type'] = 'wide'
# Last step - join df_tmp_2 with information about the numbers of beer styles



df_tmp_3 = df_beer[['brewery_id','style']].groupby(['brewery_id'], as_index=False).count()

df_finish = df_tmp_2.join(df_tmp_3.set_index('brewery_id'), on='brewery_id')
df_finish.head(10)
fig, ax = plt.subplots(1,2, figsize=(12,7))



sns.distplot(a=df_finish.ix[df_finish['width_type'] == 'single','abv'], kde=False, color='green', ax=ax[0])

sns.distplot(a=df_finish.ix[df_finish['width_type'] == 'wide','abv'], kde=False, color='red', ax=ax[0])



sns.distplot(a=df_finish.ix[df_finish['width_type'] == 'single','style'], kde=False, color='green', ax=ax[1])

sns.distplot(a=df_finish.ix[df_finish['width_type'] == 'wide','style'], kde=False, color='red', ax=ax[1])