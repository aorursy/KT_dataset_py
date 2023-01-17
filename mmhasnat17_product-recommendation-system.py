from datetime import datetime, timedelta

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M')

df = pd.read_csv('/kaggle/input/onlineretail/OnlineRetail.csv', parse_dates=['InvoiceDate'], date_parser=dateparse, encoding = 'unicode_escape')
df.head()
df.shape
df = df.loc[df['Quantity'] > 0]

df.shape
df['CustomerID'].describe()
df['CustomerID'].isna().sum()
df.loc[df['CustomerID'].isna()].head()
df.shape
df1 = df.dropna(subset=['CustomerID'])

df1.shape
df1.head()
customer_item_matrix = df1.pivot_table(index='CustomerID',columns='StockCode',values='Quantity',aggfunc='sum')
customer_item_matrix.loc[12481:].head()
customer_item_matrix.shape
df1['CustomerID'].nunique()
df1['StockCode'].nunique()
customer_item_matrix.loc[12348.0].sum()
customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x>0 else 0)
customer_item_matrix.loc[12481:].head()
from sklearn.metrics.pairwise import cosine_similarity
user_to_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
user_to_user_sim_matrix.head()
user_to_user_sim_matrix.columns = customer_item_matrix.index
user_to_user_sim_matrix['CustomerID'] = customer_item_matrix.index
user_to_user_sim_matrix = user_to_user_sim_matrix.set_index('CustomerID')
user_to_user_sim_matrix.head()
user_to_user_sim_matrix.loc[12350.0].sort_values(ascending = False)
items_bought_by_A = set(customer_item_matrix.loc[12350.0].iloc[customer_item_matrix.loc[12350.0].nonzero()].index)
items_bought_by_A
items_bought_by_B = set(customer_item_matrix.loc[17935.0].iloc[customer_item_matrix.loc[17935.0].nonzero()].index)
items_bought_by_B
items_to_recommend_User_B = items_bought_by_A - items_bought_by_B
items_to_recommend_User_B
df1.loc[

    df['StockCode'].isin(items_to_recommend_User_B),

    ['StockCode','Description']

].drop_duplicates().set_index('StockCode')
item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))
item_item_sim_matrix.columns = customer_item_matrix.T.index

item_item_sim_matrix['StockCode'] = customer_item_matrix.T.index

item_item_sim_matrix = item_item_sim_matrix.set_index('StockCode')
item_item_sim_matrix.head()
top_10_similar_items = list(

    item_item_sim_matrix\

        .loc['23166']\

        .sort_values(ascending=False)\

        .iloc[:10]\

    .index

)
top_10_similar_items
df.loc[

    df['StockCode'].isin(top_10_similar_items), 

    ['StockCode', 'Description']

].drop_duplicates().set_index('StockCode').loc[top_10_similar_items]