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
from surprise.prediction_algorithms.random_pred import NormalPredictor

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler

from surprise import accuracy

from imblearn.datasets import make_imbalance

from collections import Counter

import surprise
def LoadData():

    category_tree = pd.read_csv("../input/category_tree.csv", header= 0)

    events = pd.read_csv("../input/events.csv", header= 0)

    item_properties_part1 = pd.read_csv("../input/item_properties_part1.csv", header= 0)

    item_properties_part2 = pd.read_csv("../input/item_properties_part2.csv", header= 0)

    item_properties_part = pd.concat([item_properties_part1, item_properties_part2])

    return category_tree, events,item_properties_part

def TransfromData(category_tree, events,item_properties_part):

    data_raw = events[['visitorid','event','itemid']]

    data = data_raw.copy()

    transfrom_rating = []

    for event in data.event:

        if(event == 'view'):

            transfrom_rating.append(1)

        if(event == 'addtocart'):

            transfrom_rating.append(2)

        if(event == 'transaction'):

            transfrom_rating.append(3)

    data['rating']= transfrom_rating

    return data[['visitorid','itemid','rating']]

def RedundantData_VisistorOnlyApper(transform_data):

    data_examining = transform_data.copy()

    visitorid_size = data_examining.groupby(['visitorid']).size().reset_index(name='Size').sort_values("visitorid")

    visitorid_only_appear = visitorid_size[visitorid_size['Size']== 1]['visitorid'].tolist()

    data_surprise_remove_only_appear = data_examining[~data_examining['visitorid'].isin(visitorid_only_appear)]

    return data_surprise_remove_only_appear

def RedundantData_DropDuplicatesFeature(data_surprise_remove_only_appear):

    drop_feature = ['visitorid','itemid','rating']

    data_surprise_drop_duplicates_3_feature = data_surprise_remove_only_appear.drop_duplicates(subset=drop_feature)

    return data_surprise_drop_duplicates_3_feature

def RedundantData_SelectMaxRating(data_surprise_drop_duplicates_3_feature):

    drop_feature = ['visitorid','itemid']

    data_examining = data_surprise_drop_duplicates_3_feature.copy()

    data_seclect_max_rating = data_examining.groupby(drop_feature).max()['rating'].reset_index()

    return data_seclect_max_rating

category_tree, events,item_properties_part = LoadData()

transform_data = TransfromData(category_tree, events,item_properties_part)

data_surprise_remove_only_appear = RedundantData_VisistorOnlyApper(transform_data)

data_surprise_drop_duplicates = RedundantData_DropDuplicatesFeature(data_surprise_remove_only_appear)

data_seclect_max_rating = RedundantData_SelectMaxRating(data_surprise_drop_duplicates)
data_seclect_max_rating.head()
print("rating 1 : ",data_seclect_max_rating[data_seclect_max_rating['rating']== 1].shape)

print("rating 2 : ",data_seclect_max_rating[data_seclect_max_rating['rating']== 2].shape)

print("rating 3 : ",data_seclect_max_rating[data_seclect_max_rating['rating']== 3].shape)
data_train, data_test = train_test_split(data_seclect_max_rating, test_size = 0.25, random_state = 0)

print("data train rating 1 : ",data_train[data_train['rating']== 1].shape)

print("data train rating 2 : ",data_train[data_train['rating']== 2].shape)

print("data train rating 3 : ",data_train[data_train['rating']== 3].shape)
RANDOM_STATE = 42



number_view = data_train[data_train['rating']== 1].shape[0]

number_addtocard = data_train[data_train['rating']== 3].shape[0]

number_transaction = data_train[data_train['rating']== 2].shape[0]

implearn_feature, implearn_target = make_imbalance(data_train[['visitorid','itemid']],data_train['rating'],

                      sampling_strategy={1: int(number_addtocard*3), 2: int(number_addtocard*1.5), 3: number_addtocard},

                      random_state=RANDOM_STATE)



print('Distribution after imbalancing: {}'.format(Counter(implearn_target)))
data_surprise = pd.DataFrame(implearn_feature, columns={'visitorid','itemid'})

data_surprise['ratings ']= implearn_target

data_surprise.info()
reader = surprise.Reader(rating_scale=(1, 3))

testset_model_surprise = surprise.Dataset.load_from_df(data_test, reader).build_full_trainset()

trainset_model_surprise = surprise.Dataset.load_from_df(data_surprise, reader).build_full_trainset()

bsl_options = {'method': 'als',

               'n_epochs': 5,

               'reg_u': 12,

               'reg_i': 5

               }

algo_normal_predictor = NormalPredictor()

model_normal_predictor_surprise = algo_normal_predictor.fit(trainset_model_surprise)

testset_surprise = testset_model_surprise.build_testset()

predictions = model_normal_predictor_surprise.test(testset_surprise)
result = pd.DataFrame(predictions, columns=['visitor_id', 'item_id', 'base_event', 'predict_event', 'details'])

result.drop(columns = {'details'}, inplace = True)

result['erro'] = abs(result['base_event'] - result['predict_event'])

result.head()
print("rating 1 : ",result[result['base_event']== 1].shape)

print("rating 2 : ",result[result['base_event']== 2].shape)

print("rating 3 : ",result[result['base_event']== 3].shape)
result['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 1]['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 2]['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 3]['predict_event'].hist(bins= 100, figsize= (20,10))