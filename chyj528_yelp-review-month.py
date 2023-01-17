import os
import pickle
from time import time

from ast import literal_eval
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import kneighbors_graph
def init_ds(json):
    ds= {}
    keys = json.keys()
    for k in keys:
        ds[k]= []
    return ds, keys

def read_json(file):
    dataset = {}
    keys = []
    with open(file) as file_lines:
        for count, line in enumerate(file_lines):
            data = json.loads(line.strip())
            if count ==0:
                dataset, keys = init_ds(data)
            for k in keys:
                dataset[k].append(data[k])
                
        return pd.DataFrame(dataset)
%%time
print('loading user data...')
raw_user = read_json('../input/yelp-dataset/yelp_academic_dataset_user.json')
raw_user.head(2)
%%time
print('loading business data...')
raw_item = read_json('../input/yelp-dataset/yelp_academic_dataset_business.json').rename(columns={'business_id': 'item_id'})
raw_item.head(2)
%%time
print('loading review data...')
raw_data = read_json('../input/yelp-dataset/yelp_academic_dataset_review.json').rename(columns={'business_id': 'item_id', 'stars': 'rating'})
raw_data.head(2)
print ("shape of raw_user, raw_item, raw_data:",raw_user.shape,raw_item.shape,raw_data.shape)
user=raw_user
item=raw_item
data=raw_data
#print('user.json:',user.columns)
#print('item.json:',item.columns)
#print('data.json:',data.columns)

df = data['user_id'].value_counts()
print(df[int((len(df)-1)/2)])

print('filtering for businesses ...')
year=2009
user=raw_user
item=raw_item
data=raw_data
data['date'] = pd.to_datetime(data['date'])
item = item[item['state'].str.upper().str.replace(' ', '')  == 'ON'].reset_index(drop=True)
item_ids = dict(zip(item['item_id'], item.index))
data = data[data['item_id'].isin(item_ids) & (data['rating'] > 0) & (data['date'].dt.year == year)].reset_index(drop=True)
item = item[item['item_id'].isin(data['item_id'])].reset_index(drop=True)
item_ids = dict(zip(item['item_id'], item.index))
user = user[user['user_id'].isin(data['user_id'])].reset_index(drop=True)
user_ids = dict(zip(user['user_id'], user.index))
print('shape of user, item, data in ON', user.shape,item.shape,data.shape)
df = data['user_id'].value_counts()
print(df)
print(df[int((len(df)-1)/2)])
overlap_user=user['user_id']
overlap_user.head()
user=raw_user
item=raw_item
data=raw_data
overlap_user=user['user_id']
states=['AZ','NV','ON','OH','NC','PA','QC','AB','WI','IL','SC']
years = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
for year in years:
    user=raw_user
    item=raw_item
    data=raw_data
    data['date'] = pd.to_datetime(data['date'])
    item = item[item['state'].str.upper().str.replace(' ', '')  == states[5]].reset_index(drop=True)
    item_ids = dict(zip(item['item_id'], item.index))
    data = data[data['item_id'].isin(item_ids) & (data['rating'] > 0) & (data['date'].dt.year == year)].reset_index(drop=True)
    item = item[item['item_id'].isin(data['item_id'])].reset_index(drop=True)
    item_ids = dict(zip(item['item_id'], item.index))
    user = user[user['user_id'].isin(data['user_id'])].reset_index(drop=True)
    user_ids = dict(zip(user['user_id'], user.index))
    user_review_counts = data['user_id'].value_counts()
    print('Reviews per user in ',states[2],' in ',year,':' ,data.shape[0]/user.shape[0],'.    Median:',user_review_counts[int((len(user_review_counts)-1)/2)])
    overlap_user = pd.merge(overlap_user, user['user_id'] , on='user_id')
    print("overlap_user.shape:",overlap_user.shape)
    
    

print('filtering for businesses ...')
year=2018
user=raw_user
item=raw_item
data=raw_data
data['date'] = pd.to_datetime(data['date'])
item = item[item['state'].str.upper().str.replace(' ', '')  == 'PA'].reset_index(drop=True)
item_ids = dict(zip(item['item_id'], item.index))
data = data[data['item_id'].isin(item_ids) & (data['rating'] > 0) & (data['date'].dt.year == year)].reset_index(drop=True)
item = item[item['item_id'].isin(data['item_id'])].reset_index(drop=True)
item_ids = dict(zip(item['item_id'], item.index))
user = user[user['user_id'].isin(data['user_id'])].reset_index(drop=True)
user_ids = dict(zip(user['user_id'], user.index))
print('shape of user, item, data in ON in',year, ': ', user.shape,item.shape,data.shape,'---Reviews per user:',data.shape[0]/user.shape[0])

overlap_user=user['user_id']
data['date'] = pd.to_datetime(data['date'])
month = [1,2,3,4,5,6,7,8,9,10,11,12]
data_month = []
user_month = []
item_month = []
for i in range(12):
    data_name = 'data'+'_'+str(year)+'_'+str(i+1)
    user_name = 'user'+'_'+str(year)+'_'+str(i+1)
    item_name = 'item'+'_'+str(year)+'_'+str(i+1)
    data_month.append(data_name)
    user_month.append(user_name)
    item_month.append(item_name)
    locals()[data_name] = data[(data['date'].dt.year == year) & (data['date'].dt.month == i+1)].reset_index(drop=True)
    locals()[user_name] = user[user['user_id'].isin(locals()[data_name]['user_id'])].reset_index(drop=True)
    locals()[item_name] = item[item['item_id'].isin(locals()[data_name]['item_id'])].reset_index(drop=True)
    user_review_counts = data['user_id'].value_counts()
    print('shape of user,item，data in month ',i+1,' :',locals()[user_name].shape,locals()[item_name].shape,locals()[data_name].shape,'---Reviews per user:',locals()[data_name].shape[0]/locals()[user_name].shape[0],'.    Median:',user_review_counts[int((len(user_review_counts)-1)/2)])
    overlap_user = pd.merge(overlap_user, locals()[user_name]['user_id'] , on='user_id')
    print("overlap_user.shape:",overlap_user.shape)

    
subset_user=locals()[user_month[0]]['user_id']
for i in range(11):
    df_i = locals()[user_month[i+1]]['user_id']
    subset_user = pd.merge(subset_user, df_i, on='user_id')
print(subset_user.shape)
user=raw_user
item=raw_item
data=raw_data
states=['AZ','NV','ON','OH','NC','PA','QC','AB','WI','IL','SC']
years = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
for state in states:
    user=raw_user
    item=raw_item
    data=raw_data
    data['date'] = pd.to_datetime(data['date'])
    item = item[item['state'].str.upper().str.replace(' ', '')  == state].reset_index(drop=True)
    item_ids = dict(zip(item['item_id'], item.index))
    data = data[data['item_id'].isin(item_ids) & (data['rating'] > 0) & (data['date'].dt.year == years[3])].reset_index(drop=True)
    item = item[item['item_id'].isin(data['item_id'])].reset_index(drop=True)
    item_ids = dict(zip(item['item_id'], item.index))
    user = user[user['user_id'].isin(data['user_id'])].reset_index(drop=True)
    user_ids = dict(zip(user['user_id'], user.index))
    print('reviews per user in ',state,' in ',years[3],':' ,data.shape[0]/user.shape[0])
    
print('filtering for businesses ...')
year=2012
user=raw_user
item=raw_item
data=raw_data
data['date'] = pd.to_datetime(data['date'])
item = item[item['state'].str.upper().str.replace(' ', '')  == 'ON'].reset_index(drop=True)
item_ids = dict(zip(item['item_id'], item.index))
data = data[data['item_id'].isin(item_ids) & (data['rating'] > 0) & (data['date'].dt.year == year)].reset_index(drop=True)
item = item[item['item_id'].isin(data['item_id'])].reset_index(drop=True)
item_ids = dict(zip(item['item_id'], item.index))
user = user[user['user_id'].isin(data['user_id'])].reset_index(drop=True)
user_ids = dict(zip(user['user_id'], user.index))
print('shape of user, item, data in ON', user.shape,item.shape,data.shape)
user=raw_user
item=raw_item
data=raw_data
data['date'] = pd.to_datetime(data['date'])
year=2009
data_month = []
user_month = []
item_month = []
for i in range(10):
    data_name = 'data'+'_'+str(year+i)
    user_name = 'user'+'_'+str(year+i)
    item_name = 'item'+'_'+str(year+i)
    data_month.append(data_name)
    user_month.append(user_name)
    item_month.append(item_name)
    locals()[data_name] = data[(data['date'].dt.year == year+i)].reset_index(drop=True)
    locals()[user_name] = user[user['user_id'].isin(locals()[data_name]['user_id'])].reset_index(drop=True)
    locals()[item_name] = item[item['item_id'].isin(locals()[data_name]['item_id'])].reset_index(drop=True)
    print('shape of user,item，data in ON in year ',year+i,' :',locals()[user_name].shape,locals()[item_name].shape,locals()[data_name].shape,'---review per user:',locals()[data_name].shape[0]/locals()[user_name].shape[0])
user=raw_user
item=raw_item
data=raw_data
data['date'] = pd.to_datetime(data['date'])
year = 2018
print(year)
month = [1,2,3,4,5,6,7,8,9,10,11,12]
data_month = []
user_month = []
item_month = []
for i in range(12):
    data_name = 'data'+'_'+str(year)+'_'+str(i+1)
    user_name = 'user'+'_'+str(year)+'_'+str(i+1)
    item_name = 'item'+'_'+str(year)+'_'+str(i+1)
    data_month.append(data_name)
    user_month.append(user_name)
    item_month.append(item_name)
    locals()[data_name] = data[(data['date'].dt.year == year) & (data['date'].dt.month == i+1)].reset_index(drop=True)
    locals()[user_name] = user[user['user_id'].isin(locals()[data_name]['user_id'])].reset_index(drop=True)
    locals()[item_name] = item[item['item_id'].isin(locals()[data_name]['item_id'])].reset_index(drop=True)
    print('shape of user,item，data in month ',i+1,' :',locals()[user_name].shape,locals()[item_name].shape,locals()[data_name].shape)
    

    
subset_user=locals()[user_month[0]]['user_id']
for i in range(11):
    df_i = locals()[user_month[i+1]]['user_id']
    subset_user = pd.merge(subset_user, df_i, on='user_id')
print(subset_user.shape)

data_2018_3.head()

data_2018_3['user_id']
review_matrix = []
user_graph = []
for i in range(12):
    user_ids = dict(zip(locals()[user_month[i]]['user_id'], locals()[user_month[i]].index))
    item_ids = dict(zip(locals()[item_month[i]]['item_id'], locals()[item_month[i]].index))
    locals()[data_month[i]]['user_id'] = locals()[data_month[i]]['user_id'].apply(user_ids.get)
    locals()[data_month[i]]['item_id'] = locals()[data_month[i]]['item_id'].apply(item_ids.get)
    n_row = locals()[item_month[i]].shape[0]
    n_col = locals()[user_month[i]].shape[0]

    
    print('getting review matrix ...')
    row,col=[],[]
    row = locals()[data_month[i]]['item_id'].tolist()
    col = locals()[data_month[i]]['user_id'].tolist()
    entry = locals()[data_month[i]]['rating'].tolist()
    coo = sp.coo_matrix((entry, (row, col)), shape=(n_row, n_col))
    print (coo.toarray().shape)
    review_matrix.append(coo)
    
       
    
    print('getting user graph ...')
    f_row,f_col=[],[]
    for i, friends in locals()[user_month[i]]['friends'].items():
        for friend in friends.split(', '):
            if friend in user_ids:
                f_row.append(i)
                f_col.append(user_ids[friend])
    network=sp.coo_matrix(([1.0]*len(f_row), (f_row, f_col)), shape=(n_col,n_col))
    print (network.toarray().shape)
    user_graph.append(network)
    





