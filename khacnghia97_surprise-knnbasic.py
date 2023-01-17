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
events = pd.read_csv("../input/events.csv")

category_tree = pd.read_csv("../input/category_tree.csv")

item_properties_part1 = pd.read_csv("../input/item_properties_part1.csv")

item_properties_part2 = pd.read_csv("../input/item_properties_part2.csv")

item_properties_part = pd.concat([item_properties_part1, item_properties_part2])
events.head()
category_tree.head()
item_properties_part.head()
data = events[['visitorid','event','itemid']]

info_event_events = events.groupby(by=['event']).size()

info_event_events
import matplotlib.pyplot as plt

plt.figure()

plt.bar(['addtocart','transaction','view'],info_event_events.tolist())

#axis_2.pie(info_event_events.tolist(), labels=events['event'].unique(),autopct='%1.1f%%',shadow=True, startangle=90)

plt.title("Unique events event")

plt.show()
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
data_view  = data_surprise[data_surprise['rating']==1].reset_index(drop= True)

data_transaction  = data_surprise[data_surprise['rating']==2].reset_index(drop= True)

data_addtocard  = data_surprise[data_surprise['rating']==3].reset_index(drop= True)
from sklearn.model_selection import train_test_split as  train_test_split_sklearn

data_view_train, data_view_test = train_test_split_sklearn(data_view, test_size= 0.008)

data_transaction_train, data_transaction_test = train_test_split_sklearn(data_transaction, test_size= 0.33)
data_tuning = pd.concat([data_addtocard, data_view_test, data_transaction_test]).sort_values(by = 'rating').reset_index(drop=True)
print("The number item view ", data_tuning[data_tuning['rating']==1].shape[0])

print("The number item tranaction ", data_tuning[data_tuning['rating']==2].shape[0])

print("The number item addtacard ", data_tuning[data_tuning['rating']==3].shape[0])

data_tuning.head()
import surprise

reader = surprise.Reader(rating_scale=(1, 3))

data = surprise.Dataset.load_from_df(data_tuning, reader)

type(data)
from surprise.model_selection.split import train_test_split

trainset, testset = train_test_split(data, test_size=0.25)

type(trainset)
from surprise.prediction_algorithms.knns import KNNBasic

sim_options = {'name': 'cosine',

               'user_based': False

               }

algo_knn_basic = KNNBasic(sim_options=sim_options)

predictions = algo_knn_basic.fit(trainset).test(testset)
result = pd.DataFrame(predictions, columns=['visitor_id', 'item_id', 'base_event', 'predict_event', 'details'])

result.drop(columns = {'details'}, inplace = True)

result['erro'] = abs(result['base_event'] - result['predict_event'])

result.head()
result['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 1]['predict_event'].hist(bins= 100, figsize= (20,10))
result[result['base_event']== 2]['predict_event'].hist(bins= 100, figsize= (20,10))
from surprise import accuracy

mae_model = accuracy.mae(predictions)

rmse_model = accuracy.rmse(predictions)