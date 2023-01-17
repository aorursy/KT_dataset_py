!pip install turicreate
import os

import numpy as np

import pandas as pd

import turicreate

from zipfile import ZipFile
#zip dir

ds_dir = '../input/instacart-market-basket-analysis'
#unzip dataset

with ZipFile(os.path.join(ds_dir,"aisles.csv.zip"), 'r') as zipObj:

   zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"departments.csv.zip"), 'r') as zipObj:

   zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"order_products__prior.csv.zip"), 'r') as zipObj:

   zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"order_products__train.csv.zip"), 'r') as zipObj:

   zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"orders.csv.zip"), 'r') as zipObj:

   zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"products.csv.zip"), 'r') as zipObj:

   zipObj.extractall()

with ZipFile(os.path.join(ds_dir,"sample_submission.csv.zip"), 'r') as zipObj:

   zipObj.extractall()
#load data

prior = pd.read_csv('order_products__prior.csv')

train = pd.read_csv('order_products__train.csv')

orders = pd.read_csv('orders.csv')

products = pd.read_csv('products.csv')



#fillna

prior.fillna(0,inplace=True)

train.fillna(0,inplace=True)

orders.fillna(0,inplace=True)
#ignore prior and train, merge them all

data = pd.concat([train,prior])

ou = orders[['order_id','user_id']]



data = pd.merge(data,ou,on='order_id').drop('order_id',axis=1)
#convert reordered to rating/score

reordered = data.groupby(['user_id','product_id']).reordered.sum()

data = pd.merge(data,reordered, on=['user_id','product_id'], how='left')

data = data[data.reordered_y>0]

data['target'] = np.log(data['reordered_y']+1)

data.drop(['reordered_x','reordered_y','add_to_cart_order'],axis=1,inplace=True)
#set test dataset

test = orders[orders.eval_set=='test']

test = test[['user_id']]
#set relation dataset to exclude non recurring items

relation = data[['user_id','product_id']].copy()

relation.drop_duplicates(inplace=True)
#backup to trace nan data

sub = orders[orders.eval_set == 'test']

sub = sub[['order_id','user_id']]
#convert to turi df

train_data = turicreate.SFrame(data)

test_data = turicreate.SFrame(test)

relation_data = turicreate.SFrame(relation)
#train

recommender =  turicreate.recommender.item_similarity_recommender.create(train_data, user_id='user_id', item_id='product_id', target='target', similarity_type='jaccard', verbose=True)
def clean_prediction(row):

    data = row.products

    data = str("".join(str(data))[1:-1].replace(',',' '))

    return data
#predicting

result = recommender.recommend(users=test_data, items=relation_data, exclude_known=False, k=len(products)).to_dataframe()
thresholds = {

    'no': result.score.min()-1,

    'std' : result.score.std(),

    'zero' : 0,

    'q05' : result.score.quantile(q=0.05),

    'q10' : result.score.quantile(q=0.1)

    }



results = {}
for key,item in thresholds.items():

    print(key,':',str(item))

    results[key] = result[result['score'] > item]

    results[key] = results[key].groupby('user_id')['product_id'].apply(list).reset_index(name='products')

    results[key]['products'] = results[key].apply(clean_prediction, axis=1)
for key,item in thresholds.items():

    results[key] = pd.merge(sub, results[key],how='outer',on='user_id').sort_values('user_id')

    results[key].fillna('None', inplace=True)

    results[key].drop('user_id',axis=1,inplace=True)
for key,item in thresholds.items():

    results[key].to_csv('submission_turi_ItemSim_jaccard_pol_'+str(key)+'.csv',index=False)