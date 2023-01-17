# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import featuretools as ft

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.getcwd())



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/Train.csv")

test = pd.read_csv("../input/Test.csv")

train.head
test_item_id = test['Item_Identifier']

test_outlet_id = test['Outlet_Identifier']

sales = train['Item_Outlet_Sales']

train.drop(['Item_Outlet_Sales'], axis=1, inplace=True)
combi = train.append(test, ignore_index=True)

combi.isnull().sum()
combi['Item_Weight'].fillna(combi['Item_Weight'].mean(), inplace=True)

combi['Outlet_Size'].fillna("missing", inplace=True)
combi['Item_Fat_Content'].value_counts()
fat_content_dict = {'Low Fat':0, 'Regular':1, 'LF':0, 'reg':1, 'low fat':0}



combi['Item_Fat_Content'] = combi['Item_Fat_Content'].replace(fat_content_dict, regex=True)
combi.head()
combi['id'] = combi['Item_Identifier'] + combi['Outlet_Identifier']

combi.drop(['Item_Identifier'], axis=1, inplace=True)
# creating and entity set 'es'

es = ft.EntitySet(id = 'sales')



# adding a dataframe 

es.entity_from_dataframe(entity_id = 'bigmart', dataframe = combi, index = 'id')
es.normalize_entity(base_entity_id='bigmart', new_entity_id='outlet', index = 'Outlet_Identifier', 

additional_variables = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
feature_matrix, feature_names = ft.dfs(entityset=es, 

target_entity = 'bigmart', 

max_depth = 2, 

verbose = 1, 

n_jobs = 3)
feature_matrix.head()
feature_matrix = feature_matrix.reindex(index=combi['id'])

feature_matrix = feature_matrix.reset_index()

feature_matrix.head()
from catboost import CatBoostRegressor

categorical_features = np.where(feature_matrix.dtypes == 'object')[0]



for i in categorical_features:

    feature_matrix.iloc[:,i] = feature_matrix.iloc[:,i].astype('str')
feature_matrix.drop(['id'], axis=1, inplace=True)

train = feature_matrix[:8523]

test = feature_matrix[8523:]

train.drop(['Outlet_Identifier'], axis=1, inplace=True)

test.drop(['Outlet_Identifier'], axis=1, inplace=True)

categorical_features = np.where(train.dtypes == 'object')[0]

print(categorical_features)
from sklearn.model_selection import train_test_split



# splitting train data into training and validation set

xtrain, xvalid, ytrain, yvalid = train_test_split(train, sales, test_size=0.25, random_state=11)
model_cat = CatBoostRegressor(iterations=100, learning_rate=0.3, depth=6, eval_metric='RMSE', random_seed=7)



# training model

model_cat.fit(xtrain, ytrain, cat_features=categorical_features, use_best_model=True)

# validation score

model_cat.score(xvalid, yvalid)