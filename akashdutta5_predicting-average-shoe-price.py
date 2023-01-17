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
df=pd.read_csv("../input/womens-shoes-prices/Datafiniti_Womens_Shoes.csv")

df.head(5)

df.info()

del df['id']

del df['dateAdded']

del df['dateUpdated']

del df['asins']

del df['colors']

del df['dimension']

del df['ean']

del df['imageURLs']

del df['keys']

del df['manufacturerNumber']

del df['prices.color']

del df['prices.dateAdded']

del df['prices.dateSeen']

del df['prices.returnPolicy']

del df['prices.size']

del df['prices.sourceURLs']

del df['sourceURLs']

del df['sizes']

del df['upc']

del df['weight'] 







df.isnull().sum()
df['manufacturer'].value_counts()
df['prices.merchant'].value_counts()
df['prices.offer'].value_counts()
df['prices.shipping'].value_counts()
df.manufacturer.fillna("Dr. Scholl's" , inplace = True)

df.isnull().sum()
df.rename(columns={'prices.amountMax' : 'maxAmount' , 'prices.amountMin' : 'minAmount' ,'prices.availability': 'availability' ,  'prices.condition' : 'condition' , 'prices.currency' : 'currency' , 'prices.isSale' : 'onSale', 'prices.merchant' : 'merchant' , 'prices.offer' : 'offer' , 'prices.shipping' : 'shipping' }, inplace=True)
df.isnull().sum()
df.availability.fillna("noInfo" , inplace = True)

df.condition.fillna("noInfo" , inplace = True)

df.merchant.fillna("Backcountry.com" , inplace = True)

df.offer.fillna("25%" , inplace = True)

df.shipping.fillna("Free 2-Day shipping on orders over $50" , inplace =True)

df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt





sns.catplot(x='maxAmount' , y = 'brand' , data = df , height= 25)

plt.show()
sns.catplot(x= 'minAmount' , y = 'brand' , data = df , height = 25)

plt.show()
sns.distplot(df['maxAmount'] , bins = 8 , hist= True )
df.info()
from sklearn.preprocessing import LabelEncoder

label =LabelEncoder()

df['brand'] = label.fit_transform(df['brand'])

df['categories'] = label.fit_transform(df['categories'])

df['primaryCategories'] = label.fit_transform(df['primaryCategories'])

df['manufacturer'] = label.fit_transform(df['manufacturer'])

df['name'] = label.fit_transform(df['name'])

df['availability'] = label.fit_transform(df['availability'])

df['condition'] = label.fit_transform(df['condition'])

df['currency'] = label.fit_transform(df['currency'])

df['onSale'] = label.fit_transform(df['onSale'])

df['merchant'] = label.fit_transform(df['merchant'])

df['offer'] = label.fit_transform(df['offer'])

df['shipping'] = label.fit_transform(df['shipping'])

df.info()
df['average'] = df.apply(lambda x: x['maxAmount'] + x['minAmount'], axis=1)
from sklearn.model_selection import train_test_split

train , test = train_test_split(df , test_size = 0.2 , random_state = 1)
def data_splitting(df):

    x=df.drop(['average'], axis=1)

    y=df['average']

    return x, y

x_train , y_train = data_splitting(train)

x_test , y_test = data_splitting(test)
from sklearn.linear_model import LinearRegression

log = LinearRegression()

log.fit(x_train , y_train)
log_train = log.score(x_train , y_train)

log_test = log.score(x_test , y_test)



print("Training score :" , log_train)

print("Testing score :" , log_test)

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(x_train , y_train)
reg_train = reg.score(x_train , y_train)

reg_test = reg.score(x_test , y_test)



print("Training Score :" , reg_train)

print("Testing Score :" , reg_test)