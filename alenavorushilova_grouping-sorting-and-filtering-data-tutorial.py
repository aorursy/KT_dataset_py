import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', index_col = 'date', parse_dates = True )

df.head()
df.describe().T
df.info()
df.isnull().sum()
def summary(df):

    

    types = df.dtypes

    counts = df.apply(lambda x: x.count())

    uniques = df.apply(lambda x: [x.unique()])

    nas = df.apply(lambda x: x.isnull().sum())

    distincts = df.apply(lambda x: x.unique().shape[0])

    missing = (df.isnull().sum() / df.shape[0]) * 100

    sk = df.skew()

    krt = df.kurt()

    

    print('Data shape:', df.shape)



    cols = ['Type', 'Total count', 'Null Values', 'Distinct Values', 'Missing Ratio', 'Unique Values', 'Skewness', 'Kurtosis']

    dtls = pd.concat([types, counts, nas, distincts, missing, uniques, sk, krt], axis=1, sort=False)

  

    dtls.columns = cols

    return dtls
details = summary(df)

details
df.index.names = ['Date']

df.head()
df = df.drop(['date_block_num'], axis = 1)

df.head()
df.rename(columns={'shop_id':'Store ID', 'item_id':'Item ID', 'item_price':'Price', 'item_cnt_day':'Volume'}, inplace = True)

df.head()
df['Daily Revenue'] = df['Price']*df['Volume']

df.head()
df[['Store ID','Price']].groupby('Store ID').mean().head()

df.groupby('Store ID')[['Volume']].sum().head()
df_ml = df.groupby(['Store ID', 'Volume']).sum()

df_ml.head()
df_ml.xs(39, level = 'Store ID').head()
df_monthly = df.reset_index().groupby(['Store ID', pd.Grouper(key='Date', freq = 'M')])[['Daily Revenue']].sum().rename(columns = {'Daily Revenue':'Monthly Revenue'})

df_monthly.head()
df.reset_index().groupby(pd.Grouper(key='Date', freq='Q')).size()
df['Daily Revenue'].reset_index().groupby('Date', as_index = False).sum().rename(columns = {'Daily Revenue':'Total Daily Revenue'}).head()
grouped = df.reset_index().groupby('Store ID')

grouped.head()
type(grouped)
grouped.get_group(36).head()
grouped.size().head()
df.groupby(pd.qcut(x = df['Price'], q=3, labels=['Low', 'Medium','High'])).size()
df.groupby(pd.cut(df['Daily Revenue'], [0,500,1000,2500,5000,10000,50000, 100000, 175000, 250000, 500000, 750000, 1000000, 1250000])).size()
df.groupby(['Store ID', 'Price']).size().head(10)
df_1 = df.loc[df['Store ID']==(59)]

df_1.groupby(['Volume','Price', 'Daily Revenue']).size().head(10)
df[['Price','Volume','Daily Revenue']].agg(['sum', 'mean'])
df.agg({'Price':['mean'], 'Volume':['sum','mean'], 'Daily Revenue':['sum','mean']})
def my_agg(x): 

    names = { 

        'PriceMean': x['Price'].mean(),

        'VolumeMax': x['Volume'].max(), 

        'DailyRevMean': x['Daily Revenue'].mean(),

        'DailyRevMax': x['Daily Revenue'].max()

    }



    return pd.Series(names, index=[ key for key in names.keys()])



df.groupby('Store ID').apply(my_agg).head(10)
df.groupby('Store ID').agg({'Price':['mean'], 'Volume':['sum','mean'], 'Daily Revenue':['sum','mean']}).head()
df.groupby('Store ID').agg({'Price':['mean'], 'Volume':['sum','mean'], 'Daily Revenue':['sum','mean']}).T
df.apply(sum)
df.groupby('Store ID').apply(lambda x:x.mean()).head()
def df_mean(x):

    return x.mean()

df.groupby('Store ID').apply(df_mean).head()
df.reset_index().groupby(pd.Grouper(key = 'Date', freq = 'Q'))['Volume'].apply(sum)
df['Volume %'] = df.groupby('Store ID')[['Volume']].transform(lambda x: x/sum(x)*100)

df.head()
df.groupby('Store ID').filter(lambda x: x['Daily Revenue'].mean() > 1000).head()