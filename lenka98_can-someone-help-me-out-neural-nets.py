# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Reading the training data
train_df = pd.read_csv('../input/train.csv')
train_df.head(15)
#Columns in the training dataframe
print(train_df.columns)
#Gives a categorical mapping.
#Given a column assigns integers to the unique data of that column
def get_map(col_name):
    keys = list(train_df[col_name].unique())
    values = list(range(len(keys)))
    map_dict = dict(zip(keys, values))
    
    return map_dict
region_map = get_map('region')
region_map
city_map = get_map('city')
len(city_map)
parent_category_map = get_map('parent_category_name')
parent_category_map
category_map = get_map('category_name')
category_map
user_type_map = get_map('user_type')
user_type_map
#Adds the categorical mappings to the dataframe
def add_codes (col_name, mapping):
    train_df[col_name + '_code'] = train_df[col_name].apply(lambda x : mapping[x])
    return None
add_codes('region', region_map)
train_df[['region', 'region_code']][:10]
add_codes('city', city_map)
add_codes('parent_category_name', parent_category_map)
add_codes('category_name', category_map)
add_codes('user_type', user_type_map)
train_df[['region_code', 'city_code', 'parent_category_name_code', 'category_name_code', 'user_type_code']][:10]
#Combining all the parameters in param_1, param_2, param_3.
train_df['param_combined'] = train_df.apply(lambda row: ' '.join([str(row['param_1']), str(row['param_2']),  str(row['param_3'])]), axis=1)
train_df['param_combined'][:10]
#Finding the lenght(number of words) in a columns and adding them to the dataframe
def add_len (col_name):
    train_df[col_name] = train_df[col_name].fillna(' ')
    train_df[col_name + '_len'] = train_df[col_name].apply(lambda x : len(x.split()))
    return None
add_len('title')
train_df[['title', 'title_len']][:10]
add_len('description')
add_len('param_combined')
train_df[['title_len', 'description_len', 'param_combined_len']][:10]
#importing the periods_train.csv
pr_train_df = pd.read_csv('../input/periods_train.csv', parse_dates = ['date_from', 'date_to', 'activation_date'])
#Finding the time period for which the advertisement was there
train_df['period'] = pr_train_df['date_to'] - pr_train_df['date_from']
#Converting the time periods to number of days in integer.
train_df['period'] = train_df['period'].astype('int64')/(864 * 10e10)
train_df['period'][:10]
#Showing all the newly created columns
train_df[['region_code', 
          'city_code', 
          'parent_category_name_code', 
          'category_name_code', 
          'param_combined_len', 
          'title_len', 
          'description_len', 
          'price', 
          'user_type_code',
          'period']][:10]
#Making the training data NumPy array.
train_data = train_df[['region_code', 
                      'city_code', 
                      'parent_category_name_code', 
                      'category_name_code', 
                      'param_combined_len', 
                      'title_len', 
                      'description_len', 
                      'price', 
                      'user_type_code',
                      'period']].values
train_data.shape
train_data[:3]
#Createing the labels
labels = train_df['deal_probability'].values
labels = labels.reshape(len(labels), 1)
labels.shape
labels[:5]
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.regularizers import l2
#Implementing a simple 2-Layer Neural Net.
model = Sequential()
model.add(Dense(units = 10, 
                activation = 'relu', 
                kernel_initializer = 'glorot_uniform',
                kernel_regularizer = l2(0.01),
                input_dim = 10))
model.add(Dense(units = 1, 
                activation = 'sigmoid', 
                kernel_initializer = 'glorot_uniform', 
                kernel_regularizer = l2(0.01)))
model.compile(optimizer = Adam(lr = 0.0001,
                               beta_1 = 0.9,
                               beta_2 = 0.999,
                               epsilon = 10e-8), 
              loss = 'mean_squared_error', 
              metrics = ['mse'])
model.fit(train_data, labels, batch_size = 128, epochs = 3, validation_split = 0.01)
#Why is the acc: 0.0 and mse:inf always. Please Help!