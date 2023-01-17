# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



baseDir='/kaggle/input/19560-indian-takeaway-orders/'

orders1=pd.read_csv(str(baseDir+'restaurant-1-orders.csv'),parse_dates=['Order Date'],dayfirst=True)

orders1.columns=['order_number','date','item_name','quantity','product_price','total_products']

prices1=pd.read_csv(str(baseDir+'restaurant-1-products-price.csv'))

prices1.columns=['item_name','product_price']



orders2=pd.read_csv(str(baseDir+'restaurant-2-orders.csv'),parse_dates=['Order Date'],dayfirst=True)

orders2.columns=['order_number','date','item_name','quantity','product_price','total_products']

prices2=pd.read_csv(str(baseDir+'restaurant-2-products-price.csv'))

prices2.columns=['item_name','product_price']
OrderPrices=orders1.quantity.mul(orders1.product_price).groupby(orders1.order_number).sum().reset_index(name='total_price')

orders1=orders1.merge(OrderPrices,how='left',on='order_number')

OrderPrices=orders2.quantity.mul(orders2.product_price).groupby(orders2.order_number).sum().reset_index(name='total_price')

orders2=orders2.merge(OrderPrices,how='left',on='order_number')
ordDate1=orders1.groupby('order_number').agg(date=('date','max'),total_products=('quantity','sum' ),price=('total_price','max') )

ordDate2=orders2.groupby('order_number').agg(date=('date','max'),total_products=('quantity','sum' ),price=('total_price','max') )
plt.rcParams["figure.figsize"] = (20,10)

fig,ax=plt.subplots(1,2)

ordDate1.groupby(ordDate1.date.dt.weekday)['price'].sum().plot(kind="bar",ax=ax[0])

#ax[0].hist(ordDate1.total_products,bins=100)

ax[0].set_xlabel('Day of week')

ax[0].title.set_text('Restaurant 1')



ordDate2.groupby(ordDate2.date.dt.weekday)['price'].sum().plot(kind="bar",ax=ax[1])

#ax[1].hist(ordDate2.total_products,bins=100)

ax[1].set_xlabel('Day of week')

ax[1].title.set_text('Restaurant 2')
fig,ax=plt.subplots(1,2)

sns.boxplot(data=ordDate1, x=ordDate1.date.dt.month , y='price', ax=ax[0])

ax[0].set_ylabel('Price')

ax[0].set_xlabel('Month')

ax[0].title.set_text('Restaurant 1')



sns.boxplot(data=ordDate2, x=ordDate2.date.dt.month , y='price', ax=ax[1])

ax[1].set_ylabel('Price')

ax[1].set_xlabel('Month')

ax[1].title.set_text('Restaurant 2')

fig,ax=plt.subplots(1,2)

ax[0].plot(ordDate1.date ,ordDate1.price)

ax[0].set_ylabel('Price')

ax[0].set_xlabel('Date')

ax[0].title.set_text('Restaurant 1')



ax[1].plot(ordDate2.date ,ordDate2.price)

ax[1].set_ylabel('Price')

ax[1].set_xlabel('Date')

ax[1].title.set_text('Restaurant 2')
orderts_1=ordDate1.loc[(ordDate1.date>'2016-07-26') & (ordDate1.price<250) ]

orderts_2=ordDate2.loc[(ordDate2.date>'2016-07-26') & (ordDate2.price<250) ]
fig,ax=plt.subplots(1,2)

sns.boxplot(data=orderts_1, x=orderts_1.date.dt.hour , y='price', ax=ax[0])

ax[0].set_ylabel('Price')

ax[0].set_xlabel('Hour')

ax[0].title.set_text('Restaurant 1')



sns.boxplot(data=orderts_2, x=orderts_2.date.dt.hour , y='price', ax=ax[1])

ax[1].set_ylabel('Price')

ax[1].set_xlabel('Hour')

ax[1].title.set_text('Restaurant 2')
