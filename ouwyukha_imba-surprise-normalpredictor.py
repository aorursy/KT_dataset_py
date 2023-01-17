import os

import numpy as np

import pandas as pd

from surprise import Reader, Dataset

from zipfile import ZipFile

from surprise import NormalPredictor

from multiprocessing import Pool

from tqdm import tqdm, tqdm_pandas

tqdm_pandas(tqdm())
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

data = data[data.reordered_y>20]

data['target'] = np.log(data['reordered_y']+1)

data.drop(['reordered_x','reordered_y','add_to_cart_order'],axis=1,inplace=True)
#set test dataset

test = orders[orders.eval_set=='test']

test = test[['user_id']]
watchedList = data.groupby('user_id')['product_id'].apply(list)

itemIDs = data.product_id.unique()

userIDs = data.user_id.unique()
#backup to trace nan data

sub = orders[orders.eval_set == 'test']

sub = sub[['order_id','user_id']]
#convert to surprise df

reader = Reader(rating_scale=(data.target.min(),data.target.max()))

datatrain = Dataset.load_from_df(data, reader)

trainset = datatrain.build_full_trainset()
def predict(user):

    pred = []

    for x in itemIDs:

        try:

            if x in watchedList.loc[user.user_id]:

                pred.append((x,model.predict(user.user_id,x).est))

        except KeyError:

            continue

    pred = sorted(pred, key = lambda x: x[1], reverse=True)

    if not pred:

        return (np.nan, np.nan)

    return pred
%%time

model = NormalPredictor().fit(trainset)
num_cores = 4

num_partitions = num_cores

def parallelize_dataframe(df, func):

    df_split = np.array_split(df, num_partitions)

    pool = Pool(num_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df



def coba_mp_apply(data):

    data['predictions'] = data.progress_apply(predict, axis=1).values

    return data.explode('predictions')

    

rresults = parallelize_dataframe(test, coba_mp_apply)
rresults[['product_id', 'target']] = pd.DataFrame(rresults['predictions'].apply(pd.Series), index=rresults.index)  

rresults.drop('predictions',inplace=True,axis=1)

rresults.dropna(inplace=True)

rresults.product_id = rresults.product_id.astype(int)
thresholds = {

    'mean': rresults.target.mean(),

    'std' : rresults.target.std(),

    'zero' : 0,

    'q05' : rresults.target.quantile(q=0.05),

    'q10' : rresults.target.quantile(q=0.1)

    }



results = {}
def clean_prediction(row):

    data = row.products

    data = str("".join(str(data))[1:-1].replace(',',' '))

    return data
for key,item in thresholds.items():

    results[key] = rresults[rresults['target'] > item]

    results[key] = results[key].groupby('user_id')['product_id'].apply(list).reset_index(name='products')

    results[key]['products'] = results[key].apply(clean_prediction, axis=1)
for key,item in thresholds.items():

    results[key] = pd.merge(sub, results[key],how='outer',on='user_id').sort_values('user_id')

    results[key].fillna('None', inplace=True)

    results[key].drop('user_id',axis=1,inplace=True)
for key,item in thresholds.items():

    results[key].to_csv('submission_surprise_NormalPredictor_'+str(key)+'.csv',index=False)