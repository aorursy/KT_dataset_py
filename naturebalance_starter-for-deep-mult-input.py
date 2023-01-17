# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print('Input file:')

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", 

                    #nrows=6000000, 

                    usecols=['date','store_nbr','item_nbr','unit_sales','onpromotion'],

                    parse_dates=['date']

                   )
print('There are',str(len(train)),'rows in train.csv')

print('Columns:',str(list(train.columns)))
storeAccount=len(train.store_nbr.value_counts())

print('There are '+str(storeAccount)+' supermarket of all.')



itemAccount=len(train.item_nbr.value_counts())

print('There are '+str(itemAccount)+' items are selling.')



print('Data record from',

      str(train.date.head(1)).split( )[1],

      'to',

      str(train.date.tail(1)).split( )[1],

      '.'

     )
storeSalesVolumn=train[['store_nbr','unit_sales']].groupby('store_nbr').sum().sort_values(by='unit_sales',ascending=False)

storeSalesVolumn.unit_sales=storeSalesVolumn.unit_sales.astype('int32')



topStoreId=storeSalesVolumn.head(1).index[0]



print('Top store ID: '+str(topStoreId) )

print('sales volumn: '+str(storeSalesVolumn.head(1).values[0][0]))
itemSalesVolumn=train[['item_nbr','unit_sales']].groupby('item_nbr').sum().sort_values(by='unit_sales',ascending=False)

itemSalesVolumn.unit_sales=itemSalesVolumn.unit_sales.astype('int32')



topItemId=itemSalesVolumn.head(1).index[0]



print('Top item ID: '+str(topItemId) )

print('sales volumn: '+str(itemSalesVolumn.head(1).values[0][0]))
topStoreData=train[train['store_nbr']==topStoreId]

topStoreTopItem=topStoreData[topStoreData['item_nbr']==topItemId]



topStoreTopItem.head(70).unit_sales.plot(kind='bar',

                                          figsize=(20,5),

                                          use_index=False,

                                          xticks=list(range(0,121,7))                                        

                                        )
unWrapDate=topStoreTopItem.copy()

unWrapDate['weekday']=unWrapDate.date.dt.weekday_name

unWrapDate['month']=unWrapDate.date.dt.month

unWrapDate['year']=unWrapDate.date.dt.year

pivotWeek=unWrapDate[['year','month','weekday','unit_sales']]

pivotWeek=pivotWeek.set_index(['year','month'])

pivotWeeked=pivotWeek.pivot_table(index=['year','month'],columns='weekday',values='unit_sales')

pivotWeeked=pivotWeeked.astype('int32')

pivotWeeked=pivotWeeked[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]

pivotWeeked
import matplotlib.pyplot as plt

import seaborn as sns

step=len(pivotWeeked.index.levels[0])

plt.figure(figsize=(15,9))



for i,year in enumerate(pivotWeeked.index.levels[0]):

    plt.subplot(4, 1, i+1)

    ax = sns.heatmap(pivotWeeked.loc[year],linewidths=.5,cmap="YlGnBu")

    plt.ylabel(str(year))
transactions=pd.read_csv("../input/transactions.csv")

transactions.head()
stores=pd.read_csv("../input/stores.csv")

stores.head()
items=pd.read_csv("../input/items.csv")

items.head()
oil=pd.read_csv("../input/oil.csv")

oil.head()
holiday=pd.read_csv("../input/holidays_events.csv")

holiday.head()
from keras import layers

from keras import Input

from keras.models import Model 



#input_tensor = Input(shape=(7,))

#line = layers.Dense(32)(input_tensor) 

#line = layers.Dense(32)(line) 

#output_tensor = layers.Dense(1)(line)

#model = Model(input_tensor, output_tensor)

#model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])



weekday_input = Input(shape=(7,),  name='weekday')

tempWeekday = layers.Dense(32)(weekday_input)



month_input = Input(shape=(13,),  name='month')

tempMonth = layers.Dense(32)(month_input)



concatenated = layers.concatenate([tempWeekday, tempMonth], axis=-1)

output = layers.Dense(1)(concatenated)



myModel = Model([weekday_input, month_input], output)



myModel.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])



myModel.summary()
weekday=topStoreTopItem.date.dt.dayofweek.values

month=topStoreTopItem.date.dt.month.values

y=topStoreTopItem.unit_sales.values
from keras.utils import to_categorical

weekday=to_categorical(weekday)

month=to_categorical(month)
myModel.fit({'weekday': weekday, 'month': month},y,epochs=100, batch_size=1, verbose=1 )
[weekday[1],month[1]]
temp={'weekday': weekday, 'month': month}
predict=myModel.predict(temp)

predict[:7]