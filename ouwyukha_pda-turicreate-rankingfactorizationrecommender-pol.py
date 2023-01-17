!pip install turicreate
import os

import numpy as np

import pandas as pd

import turicreate
ds_dir = '../input/pda2019'
train = pd.read_csv(os.path.join(ds_dir,"train-PDA2019.csv"))

test = pd.read_csv(os.path.join(ds_dir,"test-PDA2019.csv"))

content = pd.read_csv(os.path.join(ds_dir,"content-PDA2019.csv"))

train.drop('timeStamp',inplace=True,axis=1)
content.drop('title',axis=1,inplace=True)

content = pd.concat([content.drop('genres', axis=1), content['genres'].str.get_dummies(sep='|')], axis=1)

content.drop('tag',axis=1,inplace=True)
content
train
test = test[['userID']]
train_data = turicreate.SFrame(train)

test_data = turicreate.SFrame(test)

content_data = turicreate.SFrame(content)
recommender =  turicreate.recommender.ranking_factorization_recommender.create(train_data, user_id='userID', item_id='itemID', target='rating', solver='ials', user_data=None, item_data=None, verbose=True)
results = recommender.recommend(users=test_data, exclude_known=True)
results = results['userID','itemID'].to_dataframe().groupby('userID')['itemID'].apply(list).reset_index(name='recommended_itemIDs')
def clean_prediction(row):

    data = row.recommended_itemIDs

    data = str("".join(str(data))[1:-1].replace(',',' '))

    return data
results['recommended_itemIDs'] = results.apply(clean_prediction, axis=1)
results
results.to_csv('sub_pda_turicreate_RankingFactorizationRecommender_pol3.csv',sep=',', index=False)