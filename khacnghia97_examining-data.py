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
category_tree = pd.read_csv("../input/category_tree.csv", header= 0)

events = pd.read_csv("../input/events.csv", header= 0)

item_properties_part1 = pd.read_csv("../input/item_properties_part1.csv", header= 0)

item_properties_part2 = pd.read_csv("../input/item_properties_part2.csv", header= 0)

item_properties_part = pd.concat([item_properties_part1, item_properties_part2])
data = events[['visitorid','event','itemid']]

info_event_events = events.groupby(by=['event']).size()

info_event_events
data.info()
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
data_surprise.sort_values(by = ['visitorid','itemid'], inplace = True)
print("total visitorid ",data_surprise['visitorid'].shape)

print("unique total visitorid ",data_surprise['visitorid'].unique().shape)
data_examining = data_surprise.copy()
data_examining.head()
visitorid_size = data_examining.groupby(['visitorid']).size().reset_index(name='Size').sort_values("visitorid")
visitorid_size.head()
visitorid_only_appear = visitorid_size[visitorid_size['Size']== 1]['visitorid'].tolist()

visitorid_size[visitorid_size['Size']== 1].shape
print("Tỉ lệ khách hàng mua một lần duy nhất ",1001560/1407580 *100,"%")
new_data_surprise = data_surprise[~data_surprise['visitorid'].isin(visitorid_only_appear)]
new_data_surprise.head()
print(" total visitorid ",2756101)

print(" new total visitori", new_data_surprise.shape[0])

print(" Tỉ lệ ", new_data_surprise.shape[0]/2756101 *100, "%")
new_data_surprise[new_data_surprise['visitorid']== 6]
new_data_surprise.shape
data_drop_duplicates = new_data_surprise.drop_duplicates(subset=['visitorid','itemid', 'rating'])

data_drop_duplicates.shape
data_drop_duplicates[data_drop_duplicates['visitorid']== 6]
print("Tỉ lệ so với các visistor xuất hiện một lần duy nhất", 1213862/1754541 *100, "%")

print("Tỉ lệ giảm tổng ", 1213862/2756101 *100, "%")
print("The number item view ", data_drop_duplicates[data_drop_duplicates['rating']==1].shape[0])

print("The number item tranaction ", data_drop_duplicates[data_drop_duplicates['rating']==2].shape[0])

print("The number item addtacard ", data_drop_duplicates[data_drop_duplicates['rating']==3].shape[0])

data_drop_duplicates.reset_index(drop=True,inplace= True)

data_drop_duplicates.head()
import surprise

from surprise import accuracy

from surprise.model_selection.split import train_test_split

from surprise.prediction_algorithms.random_pred import NormalPredictor



reader = surprise.Reader(rating_scale=(1, 3))

data_tuning = surprise.Dataset.load_from_df(data_drop_duplicates, reader)

trainset, testset = train_test_split(data_tuning, test_size=0.25)



bsl_options = {'method': 'als',

               'n_epochs': 5,

               'reg_u': 12,

               'reg_i': 5

               }

algo_normal_predictor = NormalPredictor()

predictions = algo_normal_predictor.fit(trainset)
type(predictions)
predictions.predict(uid= 0, iid= 285930, r_ui= 1)
predictions.predict(uid= 0, iid= 285930, r_ui= 1)[1]