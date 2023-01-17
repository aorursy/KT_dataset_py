import numpy as np

import pandas as pd

from collections import Counter

import json

import re



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Загружаем датасеты

train = pd.read_csv('/kaggle/input/recommendationsv4/train.csv')

test = pd.read_csv('/kaggle/input/recommendationsv4/test.csv')

submission = pd.read_csv('/kaggle/input/recommendationsv4/sample_submission.csv')



# Постройчно прочитаем json с метаданными и положим результат в датасет "meta"

with open('/kaggle/input/recommendationsv4/meta_Grocery_and_Gourmet_Food.json') as f:

    meta_list = []

    for line in f.readlines():

        meta_list.append(json.loads(line))

        

meta = pd.DataFrame(meta_list)



# Удалим дубликаты из тренировочного датасета

train.drop_duplicates(inplace = True)



# Объединим тренировочный датасет и данные из meta по идентификатору asin (Amazon Standard Identification Number)

df = pd.merge(train, meta, on='asin')

#df_new_test = pd.merge(test, meta, on='asin')
dic_verified = {

    True: 1,

    False: 0

}

df['verified'] = df['verified'].map(dic_verified)
# Заменим пропуски в main_cat на категорию "Other"

df.main_cat = df.main_cat.fillna('Other')
df.columns
features_user = df[['userid', 'verified']]

features_item = df[['itemid', 'main_cat']]

df = df[['userid','itemid','rating']]
item_f = []

col = []

unique_f1 = []

for column in features_item.drop(['itemid'], axis=1):

    col += [column]*len(features_item[column].unique())

    unique_f1 += list(features_item[column].unique())

for x,y in zip(col, unique_f1):

    res = str(x)+ ":" +str(y)

    item_f.append(res)

    print(res)
user_f = []

col = []

unique_f1 = []

for column in features_user.drop(['userid'], axis=1):

    col += [column]*len(features_user[column].unique())

    unique_f1 += list(features_user[column].unique())

for x,y in zip(col, unique_f1):

    res = str(x)+ ":" +str(y)

    user_f.append(res)

    print(res)
from lightfm.data import Dataset

# we call fit to supply userid, item id and user/item features

dataset1 = Dataset()

dataset1.fit(

        df['userid'].unique(), # all the users

        df['itemid'].unique(), # all the items

        user_features = user_f,

        item_features = item_f

)
# plugging in the interactions and their weights

(interactions, weights) = dataset1.build_interactions([(x[0], x[1], x[2]) for x in df.values ])
interactions.todense()
weights.todense()
ll = []

for column in features_item.drop(['itemid'], axis=1):

    ll.append(column + ':')

print(ll)
def feature_colon_value(my_list):

    """

    Takes as input a list and prepends the columns names to respective values in the list.

    For example: if my_list = [1,1,0,'del'],

    resultant output = ['f1:1', 'f2:1', 'f3:0', 'loc:del']



    """

    result = []

    aa = my_list

    for x,y in zip(ll,aa):

        res = str(x) +""+ str(y)

        result.append(res)

    return result
ad_subset = features_item.drop(['itemid'], axis=1)

ad_list = [x.tolist() for x in ad_subset.values]

item_feature_list = []

for item in ad_list:

    item_feature_list.append(feature_colon_value(item))

print(f'Final output: {item_feature_list[0:5]}')
item_tuple = list(zip(features_item.itemid, item_feature_list))

item_tuple[0:5]
item_features = dataset1.build_item_features(item_tuple, normalize= False)

item_features.todense()
ll = []

for column in features_user.drop(['userid'], axis=1):

    ll.append(column + ':')

print(ll)
ad_subset = features_user.drop(['userid'], axis=1)

ad_list = [x.tolist() for x in ad_subset.values]

user_feature_list = []

for user in ad_list:

    user_feature_list.append(feature_colon_value(user))

print(f'Final output: {user_feature_list[0:5]}')
user_tuple = list(zip(features_user.userid, user_feature_list))
user_features = dataset1.build_user_features(user_tuple, normalize= False)

user_features.todense()
user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()
import scipy.sparse as sparse



from lightfm import LightFM

from lightfm.cross_validation import random_train_test_split

from lightfm.evaluation import auc_score, precision_at_k, recall_at_k

import sklearn

from sklearn.model_selection import train_test_split
model = LightFM(loss='warp')

model.fit(interactions, # spase matrix representing whether user u and item i interacted

    user_features = user_features,

    item_features = item_features, # we have built the sparse matrix above

    sample_weight = weights, # spase matrix representing how much value to give to user u and item i inetraction: i.e ratings

    epochs=10)
# Довольно долго считает, так что закомментируем

# train_auc = auc_score(model,

#                       interactions,

#                       user_features = user_features,

#                       item_features=item_features

#                      ).mean()

# print('Hybrid training set AUC: %s' % train_auc)
user_ids = df.userid.apply(lambda x: user_id_map[x])

item_ids = df.itemid.apply(lambda x: item_id_map[x])

preds = model.predict(user_ids.values, item_ids.values, user_features=user_features, item_features=item_features)
sklearn.metrics.roc_auc_score(df.rating,preds)
item_feature_list = ['main_cat:Other']

user_feature_list = ['verified:1']
from scipy import sparse



def format_newitem_input(item_feature_map, item_feature_list): 

    num_features = len(item_feature_list)

    normalised_val = 1.0 

    target_indices = []

    for feature in item_feature_list:

        try:

            target_indices.append(item_feature_map[feature])

        except KeyError:

            print("new item feature encountered '{}'".format(feature))

            pass



    new_item_features = np.zeros(len(item_feature_map.keys()))

    for i in target_indices:

        new_item_features[i] = normalised_val

    new_item_features = sparse.csr_matrix(new_item_features)

    return(new_item_features)



def format_newuser_input(user_feature_map, user_feature_list):

    num_features = len(user_feature_list)

    normalised_val = 1.0 

    target_indices = []

    for feature in user_feature_list:

        try:

            target_indices.append(user_feature_map[feature])

        except KeyError:

            print("new user feature encountered '{}'".format(feature))

            pass



    new_user_features = np.zeros(len(user_feature_map.keys()))

    for i in target_indices:

        new_user_features[i] = normalised_val

    new_user_features = sparse.csr_matrix(new_user_features)

    return(new_user_features)
new_user_features = format_newuser_input(user_feature_map, user_feature_list)

preds = model.predict(0, item_ids.values, user_features=new_user_features, item_features=item_features)
new_item_features = format_newitem_input(item_feature_map, item_feature_list)

preds2 = model.predict(user_ids.values, len(user_ids.values)*[0], user_features=user_features, item_features=new_item_features)
preds3 = model.predict(0, [0], user_features=new_user_features, item_features=new_item_features)