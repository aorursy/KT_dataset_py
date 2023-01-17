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
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler

from collections import Counter
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
from imblearn.over_sampling import SMOTE, ADASYN

SMOTE_feature, SMOTE_ratings = SMOTE().fit_resample(data_train[['visitorid','itemid']],data_train['rating'])
print("SMOTE rating  : ",SMOTE_ratings.shape,Counter(SMOTE_ratings))

print("SMOTE feature : ",SMOTE_feature.shape)
feature_name = ['visitorid','itemid','rating']

data_surprise = pd.DataFrame(SMOTE_feature, columns={'visitorid','itemid'})

data_surprise['ratings ']= SMOTE_ratings

data_surprise.info()
from surprise import accuracy

from surprise.model_selection.split import train_test_split

from surprise.prediction_algorithms.random_pred import NormalPredictor

import surprise



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
train_to_testset = trainset_model_surprise.build_testset()

new_predictions = model_normal_predictor_surprise.test(train_to_testset)



new_result = pd.DataFrame(new_predictions, columns=['visitor_id', 'item_id', 'base_event', 'predict_event', 'details'])

new_result.drop(columns = {'details'}, inplace = True)

new_result['erro'] = abs(new_result['base_event'] - new_result['predict_event'])

new_result.head()
new_result['predict_event'].hist(bins= 100, figsize= (20,10))
new_result[new_result['base_event']== 1]['predict_event'].hist(bins= 100, figsize= (20,10))
new_result[new_result['base_event']== 2]['predict_event'].hist(bins= 100, figsize= (20,10))
new_result[new_result['base_event']== 3]['predict_event'].hist(bins= 100, figsize= (20,10))
from imblearn.over_sampling import SMOTE, ADASYN

ADASYN_feature, ADASYN_ratings = ADASYN().fit_resample(data_train[['visitorid','itemid']],data_train['rating'])
print("ADASYN rating  : ",ADASYN_ratings.shape,Counter(SMOTE_ratings))

print("ADASYN feature : ",ADASYN_feature.shape)
ADASYN_data_surprise = pd.DataFrame(ADASYN_feature, columns={'visitorid','itemid'})

ADASYN_data_surprise['ratings ']= ADASYN_ratings

ADASYN_data_surprise.info()
ADASYN_testset_model_surprise = surprise.Dataset.load_from_df(data_test, reader).build_full_trainset()

ADASYN_trainset_model_surprise = surprise.Dataset.load_from_df(ADASYN_data_surprise, reader).build_full_trainset()



ADASYN_algo_normal_predictor = NormalPredictor()

ADASYN_model_normal_predictor_surprise = ADASYN_algo_normal_predictor.fit(ADASYN_trainset_model_surprise)

ADASYN_testset_surprise = ADASYN_testset_model_surprise.build_testset()

ADASYN_predictions = ADASYN_model_normal_predictor_surprise.test(ADASYN_testset_surprise)
ADASYN_result = pd.DataFrame(ADASYN_predictions, columns=['visitor_id', 'item_id', 'base_event', 'predict_event', 'details'])

ADASYN_result.drop(columns = {'details'}, inplace = True)

ADASYN_result['erro'] = abs(ADASYN_result['base_event'] - ADASYN_result['predict_event'])

ADASYN_result.head()
ADASYN_result['predict_event'].hist(bins= 100, figsize= (20,10))
ADASYN_result[ADASYN_result['base_event']== 1]['predict_event'].hist(bins= 100, figsize= (20,10))
ADASYN_result[ADASYN_result['base_event']== 2]['predict_event'].hist(bins= 100, figsize= (20,10))
ADASYN_result[ADASYN_result['base_event']== 3]['predict_event'].hist(bins= 100, figsize= (20,10))