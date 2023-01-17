import numpy as np

import pandas as pd

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

path = '/kaggle/input/app-store-apple-data-set-10k-apps'

os.chdir(path)

app_data = pd.read_csv('AppleStore.csv')

app_data.head()
app_data.describe().T
app_data = pd.read_csv('AppleStore.csv')

app_data_rating = app_data.sort_values('user_rating',ascending=False)[['track_name','user_rating','price']]

app_data_rating[:10]
app_data = pd.read_csv('AppleStore.csv')

app_data_rating = app_data.sort_values('user_rating',ascending=False)[['track_name','user_rating','price']]

app_data_rating[-10:]
app_data_price = app_data.sort_values('price',ascending=False)[['track_name','user_rating','price']]

app_data_price[:10]
app_data_price_hist = app_data[['price']]

app_data_price_hist.plot.hist(bins=100)
app_data_rating_hist = app_data[['user_rating']]

app_data_rating_hist.plot.hist(bins=10)
app_data_cate_bar = app_data.groupby(['prime_genre'])[['id']].count().reset_index().sort_values('id',ascending=False)

app_data_cate_bar.columns = ['prime_genre','Nbr']

top_categories = app_data_cate_bar.head(10)



sns.barplot(y = 'prime_genre',x = 'Nbr', data=top_categories)

top_categories
top_categories = app_data_cate_bar.tail(10)

sns.barplot(y = 'prime_genre',x = 'Nbr', data=top_categories)

top_categories