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
from surprise import accuracy

from surprise.model_selection.split import train_test_split

from surprise.prediction_algorithms.random_pred import NormalPredictor

import surprise
category_tree = pd.read_csv("../input/category_tree.csv", header= 0)

events = pd.read_csv("../input/events.csv", header= 0)

item_properties_part1 = pd.read_csv("../input/item_properties_part1.csv", header= 0)

item_properties_part2 = pd.read_csv("../input/item_properties_part2.csv", header= 0)

item_properties_part = pd.concat([item_properties_part1, item_properties_part2])
data = events[['visitorid','event','itemid']]

data.head()
transfrom_rating = []

# view = 1, addtocart = 2, transaction = 3

def transfrom_data(data_raw):

    data = data_raw.copy()

    for event in data.event:

        if(event == 'view'):

            transfrom_rating.append(1)

        if(event == 'addtocart'):

            transfrom_rating.append(2)

        if(event == 'transaction'):

            transfrom_rating.append(3)

    data['rating']= transfrom_rating

    return data[['visitorid','itemid','rating']]

data_surprise = transfrom_data(data)

data_surprise.head()
reader = surprise.Reader(rating_scale=(1, 3))

data_tuning = surprise.Dataset.load_from_df(data_surprise, reader)

type(data_tuning)
bsl_options = {'method': 'als',

               'n_epochs': 5,

               'reg_u': 12,

               'reg_i': 5

               }

trainset, testset = train_test_split(data_tuning, test_size=0.25)

algo_normal_predictor = NormalPredictor()

predictions = algo_normal_predictor.fit(trainset).test(testset)
result = pd.DataFrame(predictions, columns=['visitor_id', 'item_id', 'base_event', 'predict_event', 'details'])

result.drop(columns = {'details'}, inplace = True)

result['erro'] = abs(result['base_event'] - result['predict_event'])

result.head()
result['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 1]['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 2]['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 3]['predict_event'].hist(bins= 100, figsize= (20,10))
print("The number item view ", data_surprise[data_surprise['rating']==1].shape[0])

print("The number item tranaction ", data_surprise[data_surprise['rating']==2].shape[0])

print("The number item addtacard ", data_surprise[data_surprise['rating']==3].shape[0])

data_surprise.head()
data_view  = data_surprise[data_surprise['rating']==1].reset_index(drop= True)

data_transaction  = data_surprise[data_surprise['rating']==2].reset_index(drop= True)

data_addtocard  = data_surprise[data_surprise['rating']==3].reset_index(drop= True)
data_addtocard.shape
from sklearn.model_selection import train_test_split as  train_test_split_sklearn

data_view_train, data_view_test = train_test_split_sklearn(data_view, test_size= 0.008)

data_transaction_train, data_transaction_test = train_test_split_sklearn(data_transaction, test_size= 0.33)
new_data_tuning = pd.concat([data_addtocard, data_view_test, data_transaction_test]).sort_values(by = 'rating').reset_index(drop=True)
print("The number item view ", new_data_tuning[new_data_tuning['rating']==1].shape[0])

print("The number item tranaction ", new_data_tuning[new_data_tuning['rating']==2].shape[0])

print("The number item addtacard ", new_data_tuning[new_data_tuning['rating']==3].shape[0])

print("The new data train ", new_data_tuning.shape)

new_data_tuning.head()
reader = surprise.Reader(rating_scale=(1, 3))

new_data_train= surprise.Dataset.load_from_df(new_data_tuning, reader)

type(new_data_train)
bsl_options = {'method': 'als',

               'n_epochs': 5,

               'reg_u': 12,

               'reg_i': 5

               }

train, test = train_test_split(new_data_train, test_size=0.25)

algo_normal_predictor_fix_rating = NormalPredictor()

predictions_fix_rating = algo_normal_predictor_fix_rating.fit(train).test(test)
result_fix_rating = pd.DataFrame(predictions_fix_rating, columns=['visitor_id', 'item_id', 'base_event', 'predict_event', 'details'])

result_fix_rating.drop(columns = {'details'}, inplace = True)

result_fix_rating['erro'] = abs(result_fix_rating['base_event'] - result_fix_rating['predict_event'])

result_fix_rating.head()
result_fix_rating['predict_event'].hist(bins= 100, figsize= (20,10))
result_fix_rating[result_fix_rating['base_event']== 1]['predict_event'].hist(bins= 100, figsize= (20,10))
result_fix_rating[result_fix_rating['base_event']== 2]['predict_event'].hist(bins= 100, figsize= (20,10))
result_fix_rating[result_fix_rating['base_event']== 3]['predict_event'].hist(bins= 100, figsize= (20,10))
result_fix_rating['erro'].hist(bins= 100, figsize= (20,10))
from surprise import accuracy

mae_model = accuracy.mae(predictions_fix_rating)

rmse_model = accuracy.rmse(predictions_fix_rating)