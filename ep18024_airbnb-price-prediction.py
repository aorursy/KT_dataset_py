# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/train.csv', index_col=0)

train_df
test_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/test.csv', index_col=0)

test_df
df = pd.concat([train_df.drop('price', axis=1), test_df])

df
df = pd.concat([df, pd.get_dummies(df['neighbourhood_group'])], axis=1)

df = df.drop('neighbourhood_group', axis=1)

df
df = pd.concat([df, pd.get_dummies(df['neighbourhood'])], axis=1)

df = df.drop('neighbourhood', axis=1)

df
df = pd.concat([df, pd.get_dummies(df['room_type'])], axis=1)

df = df.drop('room_type', axis=1)

df
df = df.drop(['name', 'host_id', 'host_name','last_review', 'reviews_per_month'], axis=1)

df
nrow, ncol = train_df.shape

price_df = train_df[['price']]

train_df = df[:nrow]

train_df = pd.concat([train_df, price_df], axis=1)

train_df
nrow, ncol = train_df.shape

test_df = df[nrow:]

test_df
import lightgbm as lgb

X = train_df.drop(['price'], axis=1).to_numpy()

y = train_df['price'].to_numpy()

params = {                                                                                               

    'boosting_type': 'gbdt',                                                                             

    'objective': 'regression_l2',                                                                           

    'metric': 'l2', 

    'num_leaves': 35,        #??????????????????

    'num_iterations':120,    #?????????

    'learning_rate': 0.05,   #?????????                                                                            

    'feature_fraction': 1.0,   #???????????????????????? *100%????????????                                                                           

    'bagging_fraction': 0.8,                                                                             

    'bagging_freq': 5,

     'lambda_l2': 2

}     

train_data_set = lgb.Dataset(X, y)

gbm =  lgb.train(params,train_data_set)

X = test_df.to_numpy()



p = gbm.predict(X)
submit_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')

submit_df['price'] = p

submit_df.to_csv('submission.csv', index=False)