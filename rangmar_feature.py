import pandas as pd

import numpy as np

import warnings 

warnings.filterwarnings('ignore')



train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')



y_train = train['price']

x_train = train.drop(['id'], axis=1)

x_test = test.drop(['id'], axis=1)



train_len = len(x_train)

df = pd.concat([x_train, x_test], axis=0)

df = df.drop(['date'], axis=1)
df['sqft_total_size'] = df['sqft_above'] + df['sqft_basement']

df['total_rooms'] = df['bedrooms'] * df['bathrooms']
df['per_price'] = df['price']/df['sqft_total_size']

zipcode_price = df.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

df = pd.merge(df, zipcode_price,how='left',on='zipcode')



del df['per_price']



df['price_per_bathrooms'] = df['price'] / (df['bathrooms']+0.01)

zipcode_bathrooms_price = df.groupby(['zipcode'])['price_per_bathrooms'].agg(['mean']).reset_index()

df= pd.merge(df, zipcode_bathrooms_price, how='left', on='zipcode')



del df['price_per_bathrooms']