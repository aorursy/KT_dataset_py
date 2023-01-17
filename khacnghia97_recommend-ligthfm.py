# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from lightfm import LightFM

from lightfm.data import Dataset

from sklearn.model_selection import train_test_split

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
data_seclect_max_rating.head()
data_seclect_max_rating.info()
data = data_seclect_max_rating.copy()
data.head()
def ProcessData(data_form_pandas):

    data_form_pandas.sort_values(by =['visitorid','itemid'], inplace = True)

    data_form_pandas.reset_index(drop=True, inplace = True)

    return data_form_pandas

def Onehot(list_sample):

    if(list_sample != 0):

        return 1

    else:

        return 0

def CountSame(frist_list, second_list):

    return list(set(frist_list).intersection(set(second_list)))

def InteractionTransform(transform_data):

    data_numpy = np.array(transform_data)

    data_lightfm = Dataset()

    data_lightfm.fit(users= transform_data['visitorid'].unique(),items= transform_data['itemid'].unique())

    interactions, weigths = data_lightfm.build_interactions(

        (feature[0],feature[1],feature[2]) for feature in data_numpy)

    return interactions, weigths

def MergeInteraction(original_data):

    interaction, weight = InteractionTransform(original_data)

    interaction_user = interaction.row

    interaction_item = interaction.col

    original_data['transform user'] = interaction_user

    original_data['transform item'] = interaction_item

    query_data = original_data.copy()

    return query_data

def BuildLightFmModel(data_form_pandas):

    data = data_form_pandas.copy()

    model = LightFM(loss='warp')

    interaction, weight = InteractionTransform(data)

    model.fit_partial(interactions= interaction, sample_weight= weight)

    return model
data_train, data_test = train_test_split(data, test_size = 0.25)
data_test.head()
dataset = data_test.copy()
data_tuning = ProcessData(dataset)

interactions, weights = InteractionTransform(data_tuning)

query_data = MergeInteraction(data_tuning)

model_lightfm = BuildLightFmModel(data_tuning)
query_data.head()
model_lightfm
def PredictModel(lightfm_model, data_model, user_ids, verbose= False):

    query_data = MergeInteraction(data_model)

    query_user = query_data[query_data['visitorid'].isin(user_ids)]['transform user'].unique()

    count = 0

    original_item = []

    recommend_item = []

    item_same = []

    length_item_same = []

    for user_id in query_user:

        known_item = query_data[query_data['transform user'] == user_id]['itemid'].tolist()

        item_for_user = query_data['transform item'].unique().tolist()

        scores = lightfm_model.predict([user_id], item_for_user)

        top_items = query_data['itemid'][np.argsort(-scores)].tolist()

        if(verbose == True):

            print("User %s" % user_ids[count])

            print("     Known positives: ",known_item[:5])

            print("     Recommended: ",top_items[:5])

        count+= 1

        original_item.append(known_item[:5])

        recommend_item.append(top_items[:5])

        item_same.append(CountSame(known_item[:5], top_items[:5]))

        length_item_same.append(len(CountSame(known_item[:5], top_items[:5])))

        

    recommend = pd.DataFrame(user_ids, columns={'visitorid'})

    recommend['original item'] = original_item

    recommend['recommend item'] = recommend_item

    recommend['item same'] = item_same

    recommend['length'] = length_item_same

    recommend['one hot'] = recommend['length'].apply(lambda x: Onehot(x))

    return recommend       

def Accuracy(recommend_form_pandas):

    positive_user = recommend_form_pandas[recommend_form_pandas['length'] != 0].shape[0]

    return positive_user/recommend_form_pandas.shape[0]
user = dataset['visitorid'].unique().tolist()

user[:5]
user = query_data['visitorid'].unique().tolist()

user[:5]
recommend_data = PredictModel(model_lightfm, dataset, user[:500])
recommend_data.head()
recommend_data[recommend_data['length'] != 0]
print("ACC : ",Accuracy(recommend_data))