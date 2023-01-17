import pandas as pd

import numpy as np

from datetime import datetime

import pandas.io.sql as psql

import matplotlib.pyplot as plt

import os

%matplotlib notebook
pd.set_option('display.expand_frame_repr', False)

pd.set_option('display.max_columns', None)  

pd.set_option('display.max_colwidth', -1)

pd.options.display.float_format = '{:,.2f}'.format
!ls ../input/wine-reviews
inpath = '../input/wine-reviews/'
dfcsv = pd.read_csv(inpath + 'winemag-data-130k-v2.csv')
dfcsv2 = pd.read_csv(inpath + 'winemag-data_first150k.csv') 
dfcsv2.head()
dfcsv.head() #returns top 5
dfcsv.tail() #returns bottom 5
dfcsv.columns
dfcsv.shape
dfcsv.dtypes
dfcsv.info()
dfcsv.describe()
dfcsv.columns
dfcsv[['country', 'description', 'designation']]
dfcsv.head()
dfcsv[dfcsv.country == 'Italy']
dfcsv[(dfcsv.country == 'Italy') | (dfcsv.region_1 == 'Etna') ]
dfcsv[(dfcsv.country == 'Italy') & (dfcsv.region_1 == 'Etna') ]
dfcsv[(dfcsv.country == 'Italy') & (dfcsv.region_1 == 'Etna')][['country', 'region_1', 'designation']]
dfcsv.head()
dfcsv[dfcsv.taster_name.duplicated(keep=False)]
dfcsv.head()
dfcsv.sort_values(by =['country'] , ascending = True)
dfcsv.sort_values(by =['points', 'country' ] , ascending = [False,True])
dfcsv.head()
dfcsv['points_price'] = dfcsv.points / dfcsv.price  
dfcsv[['country', 'points','price','points_price', ]].head()
def priceClass(price):

    if price > 30:

        rclass = 'Expensive'

    elif price <= 30:

        rclass = 'Cheap'

    else:

        rclass = 'Error'

    return rclass
dfcsv['priceClass'] = dfcsv.price.apply(priceClass)
dfcsv[['price','priceClass']]
fname = lambda point: 'Good Quality' if (point >60 ) else 'Bad Quality' 
dfcsv['Quality'] = dfcsv.points.apply(fname)
dfcsv[['points', 'Quality']]
def multiCol(pClass,Qual):

    if pClass == 'Cheap' and  Qual == 'Good Quality':

        stat = 'Good Buy'

    else:

        stat = 'Dont Buy'

    

    return stat
dfcsv['Buy_NotBuy'] = dfcsv.apply(lambda df: multiCol(df.priceClass, df.Quality), axis = 1) 
dfcsv.head() #ahhh Good buy == Goodbye ahhh :')
dfcsv.country.value_counts()
dfcsv.groupby('country').sum()
dfcsv.groupby('country').aggregate({'points':np.sum,'price':np.mean}).sort_values(by = 'points', ascending = False)
dfcsv.head()
pd.pivot_table(dfcsv, index = 'country', aggfunc={'points':np.sum,'price':np.mean}).sort_values(by = 'points', ascending = False)
pd.pivot_table(dfcsv, values =['points', 'price'] , index = 'taster_name', columns = 'country', aggfunc={'points':np.sum,'price':np.mean})
dfcsv.to_csv(inpath + '/wine.csv') 
dfcsv.to_pickle(inpath + '/wine.pickle')