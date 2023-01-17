# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

from datetime import datetime

from pandasql import sqldf

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_path = '/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv'

prices = pd.read_csv(data_path)

prices.head(15

         )



prices.head(7)
prices.tail(4)
p=prices['Date'].value_counts()

p

prices =pd.DataFrame(prices)



#this will be used to find missing date 

p = prices['Date'].isin(['ug 18 2016'])

prices[p]
#this will be used to correct the date error

prices.at[143,'Date'] = 'Aug 18 2016'

prices.iloc[143]


prices['month'] = pd.DatetimeIndex(prices['Date']).month_name()

prices


c_month=prices.groupby('month').size().reset_index(name='Number of Records')

c_month
prices['Year'] = pd.DatetimeIndex(prices['Date']).year

prices
prices['day'] = pd.DatetimeIndex(prices['Date']).day

prices['timestamp'] = pd.DatetimeIndex(prices['Date']).time

prices
prices2=prices[['Date','Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery']]

prices2['datetime']= pd.to_datetime(prices['Date'])

prices2
a = """Select * from prices2 where datetime between '2018-10-01' and '2019-12-30' order by datetime asc;"""

pricesoutcome = sqldf(a, globals())

pricesoutcome
pricesoutcome.plot(kind="line",x='Date', title="Graph representing Gas Prices over 8 month ", figsize=(12,8))

HFO = pd.DataFrame(pricesoutcome['HFO'])

HFO.pct_change(periods=4)
HFO.pct_change(periods=4).plot()

pkm = prices[['day','month','Year','timestamp','HFO','Butane']]

pkm
pkm.plot(kind = 'line',x = 'HFO', y='Butane')
#this is to find the  missing values

print( pkm.isnull().sum())
#this is to try and replace the missing values

pkm.fillna(-1, inplace=True)

pkm
pv = pkm[['HFO','Butane']]

pv = pv.iloc[:,:].values

pv


K_Means = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=250) 

pkm["cluster"] = K_Means.fit_predict( pv )

pkm.tail(25)
pkm['cluster'].value_counts()
pkm['cluster'].describe()
baby = pd.DataFrame(pkm['cluster'].value_counts())



sns.pairplot( pkm, hue="cluster")
prices.describe()
#showing the average price per year of each gas type before the clustering of data 

#prices.groupby(Year). Mean()



average=prices.groupby('Year').mean()

average