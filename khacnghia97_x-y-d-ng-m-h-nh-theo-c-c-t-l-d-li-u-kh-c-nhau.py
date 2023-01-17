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
data_tuning = data_seclect_max_rating.copy()

data_tuning.info()
import surprise

from surprise import accuracy

from surprise.model_selection.split import train_test_split

from surprise.prediction_algorithms.random_pred import NormalPredictor



reader = surprise.Reader(rating_scale=(1, 3))

data_tuning_model= surprise.Dataset.load_from_df(data_tuning, reader)

trainset, testset = train_test_split(data_tuning_model, test_size=0.55)



bsl_options = {'method': 'als',

               'n_epochs': 5,

               'reg_u': 12,

               'reg_i': 5

               }

algo_normal_predictor = NormalPredictor()

predictions = algo_normal_predictor.fit(trainset)
testset = trainset.build_testset()

predictions_train = algo_normal_predictor.test(testset)

result_train = pd.DataFrame(predictions_train, columns=['visitor_id', 'item_id', 'base_event', 'predict_event', 'details'])

result_train.drop(columns = {'details'}, inplace = True)

result_train['erro'] = abs(result_train['base_event'] - result_train['predict_event'])

result_train.head()
from surprise import accuracy

print("RMSE : ",accuracy.rmse(predictions_train, verbose= False))

print("MEA : ",accuracy.mae(predictions_train, verbose= False))
testset
result_train['predict_event'].hist(bins= 100, figsize= (20,10))
result_train[result_train['base_event']== 1]['predict_event'].hist(bins= 100, figsize= (20,10))
result_train[result_train['base_event']== 2]['predict_event'].hist(bins= 100, figsize= (20,10))
result_train[result_train['base_event']== 3]['predict_event'].hist(bins= 100, figsize= (20,10))
list_split = np.linspace(0.1,0.9,10)

def PlotAccuracy(list_split):

    list_rmse = []

    list_mae = []

    reader = surprise.Reader(rating_scale=(1, 3))

    data_tuning_model= surprise.Dataset.load_from_df(data_tuning, reader)

    bsl_options = {'method': 'als',

                   'n_epochs': 5,

                   'reg_u': 12,

                   'reg_i': 5

                  }

    for test_size in list_split:

        trainset, testset = train_test_split(data_tuning_model, test_size= test_size)

        algo_normal_predictor = NormalPredictor()

        predictions = algo_normal_predictor.fit(trainset)

        testset = trainset.build_testset()

        predictions_train = algo_normal_predictor.test(testset)

        list_rmse.append(accuracy.rmse(predictions_train, verbose= False))

        list_mae.append(accuracy.mae(predictions_train, verbose= False))

    return list_rmse, list_mae



list_rmse, list_mae = PlotAccuracy(list_split)
import matplotlib.pyplot as plt

plt.figure(figsize = (15,7))

plt.plot(list_split, list_rmse)

plt.plot(list_split, list_rmse, 'o')

plt.plot(list_split, list_mae)

plt.plot(list_split, list_mae, '*')