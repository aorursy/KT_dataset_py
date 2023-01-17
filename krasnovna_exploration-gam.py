!pip -q install pyLDAvis

!python3 -q -m spacy download en

!pip -q install pygam
import os

import matplotlib.pyplot as plt

from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input as resnet50_preprocess, decode_predictions



import numpy as np

import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer

from tqdm import tqdm



import gensim.corpora as corpora

import gensim

from gensim.utils import lemmatize, simple_preprocess

import time

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD

import re

from pprint import pprint

import spacy

from sklearn.metrics import silhouette_score, mean_squared_error

from datetime import datetime

import matplotlib.pyplot as plt



from collections import Counter

import seaborn as sns



####Ignoring warnings

import warnings

warnings.filterwarnings('ignore')



from pprint import pprint

from scipy.sparse import csc_matrix

from scipy.sparse.linalg import svds, eigs



import cv2
pd.read_csv('../input/tmdb-box-office-prediction/test.csv').shape
input_data = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')

full_data = pd.concat([input_data, test])
from sklearn.model_selection import train_test_split, KFold

from sklearn.cluster import KMeans

kfold = KFold(n_splits = 5, random_state = 32)

train_id = []

val_id = []

for (x1,x2) in kfold.split(np.array(np.arange(input_data.shape[0]), ndmin = 2 ).T):

    train_id.append(x1)

    val_id.append(x2)
input_data['log_revenue'] = np.log(input_data['revenue'])

log_rev_distribution = sns.boxplot(input_data['log_revenue'].values)
Q1 = input_data['log_revenue'].quantile(0.25)

Q3 = input_data['log_revenue'].quantile(0.75)

IQR = Q3 - Q1

print(IQR)

print('amount of outliers: {}'.format(

input_data[(input_data['log_revenue'] < (Q1 - 1.5 * IQR)) | (input_data['log_revenue'] > (Q3 + 1.5 * IQR) ) ].shape[0]

) )
def var_creation(df, flag_train = True):

  

    df['release_year'] = df['release_date'].fillna('01/01/00').apply( lambda x: int('19' + x.split('/')[-1]) if int(x.split('/')[-1]) > 20 else int('20' + x.split('/')[-1]) ) 

    df['years_to_release'] = 2020 - df['release_year']

    df['release_month'] = df['release_date'].fillna('01/01/00').apply( lambda x: int(x.split('/')[0]) )

    df['release_day'] = df['release_date'].fillna('01/01/00').apply( lambda x: int(x.split('/')[1]) )

    df['flag_release_season'] =  df['release_month'].apply(lambda x: x//3  )



    df['runtime'].fillna(np.median(df['runtime'].dropna()), inplace = True)





    df['budget'].fillna(np.median(df['budget'].dropna()), inplace = True)

    #replace seros with mean 

    df['budget'].replace(0,np.median(df['budget'].dropna()), inplace = True)



    df['log_budget'] = np.log(df['budget'])

    df['flag_en'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0 )

    df['release_dt'] = df.apply(lambda x: datetime(x['release_year'], x['release_month'], x['release_day']), axis = 1)



    df['crew_director'] = df['crew'].fillna('').apply(lambda x : ''.join([item['name'] for item in eval(x) if  item['job'] == 'Director'  ][:1]) if  x != '' else '' )

    df['release_dt'] = df.apply(lambda x: datetime(x['release_year'], x['release_month'], x['release_day']), axis = 1)





    df['genre_list'] = df['genres'].fillna('').apply(lambda x: [w['name'] for w in eval(x)  ] if  x != '' else [])

    df['country_list'] = df['production_countries'].fillna('').apply(lambda x : [item['name'] for item in eval(x) ] if  x != '' else [] )

    df['keywords_list'] =  df['Keywords'].fillna('').apply(lambda x : [item['name'] for item in eval(x)] if  x != '' else [] )

    df['crew_list'] =  df['crew'].fillna('').apply(lambda x : [item['name'] for item in eval(x)] if  x != '' else [] )

    df['cast_list'] =  df['cast'].fillna('').apply(lambda x : [item['name'] for item in eval(x)] if  x != '' else [] )

    df['prod_list'] =  df['production_companies'].fillna('').apply(lambda x : [item['name'] for item in eval(x)] if  x != '' else [] )

    df['overview_fst'] = df['overview'].fillna('').apply(lambda x: ' '.join(x.split(' ')[:50]) )



    if flag_train == True:

        df['log_revenue'] = np.log(df['revenue'] )



    return df





input_data = var_creation(input_data,flag_train = True)

test = var_creation(test,flag_train = False)

full_data = var_creation(full_data,flag_train = True)

def get_emb_by_pmi(text, column, top_w, flag_svd = False ):

  

 

  text_item_dict = sorted(dict(Counter(text.sum())).items(), key = lambda x: x[1], reverse = True)



  df = pd.DataFrame(text_item_dict)

  df['rate'] = df[1].cumsum()/df[1].sum()



  item_to_idx = dict([(item, idx) for (idx, item) in enumerate(df[df['rate'] <= top_w][0].values)])

  idx_to_item = dict([(idx, item) for (idx, item) in enumerate(df[df['rate'] <= top_w][0].values)])



  item_df = pd.DataFrame( data = np.stack(text.apply( lambda x: [1 if item in x  else 0 for (item, c) in item_to_idx.items()] ).values),

               columns =  [item for (item, _) in item_to_idx.items()] )

  imem_df = item_df[item_df.sum(axis = 1) >0 ]



  

  N = item_df.shape[0]

  V = len(item_to_idx)

  add = 1

  # V = 0

  pmi_matrix = []

  for (item, _) in item_to_idx.items():

    pmi_matrix.append( item_df[item_df[item] == 1].sum().values )



  pmi_matrix = (( np.stack(pmi_matrix) + add ) /( N+V) )/ np.dot( np.array( ( item_df.sum(axis = 0 ).values + add ) / (N+V) , ndmin = 2 ).T,

         np.array(( item_df.sum(axis = 0 ).values + add ) / (N+V), ndmin = 2 )

          )



  for i in range(pmi_matrix.shape[0]):

    for j in range(pmi_matrix.shape[0]):

      pmi_matrix[i][j] = 0 if pmi_matrix[i][j] < 1 else np.log(pmi_matrix[i][j])



  #using pmi_matrix as emb vectors 0 for unknown genres

  emb_genre_to_idx = dict([(item,item_to_idx[item]+1) for item, _ in item_to_idx.items() ])

  emb_matrix = np.vstack( [np.mean(pmi_matrix,axis =0),

      pmi_matrix])

  

  if flag_svd:

#     svd = TruncatedSVD(n_components = 100, random_state = 42)

    m = csc_matrix(emb_matrix, dtype=float)

    u, s, vt = svds(m, k=100)

    emb_matrix = u*s

#     print('SVD explained ratio', sum(svd.explained_variance_ratio_) )

    

  return item_to_idx, emb_matrix



# column = 'genres'

# top_w = 1

# i = 1 

# train = pd.merge( pd.DataFrame( data = train_id[1].reshape(-1), columns = ['id']), input_data )

# get_emb_by_pmi(train, column, top_w )



def fit_pmi_cluster(inp, column, top_w, nums_cluster = [2,4,8,16,32,64],flag_svd = False):

  

  base_rmse = {}

  train_rmse = {}

  val_rmse = {}

  sil_score = {}

    

  for k in nums_cluster :

    

    base_rmse[k] = []

    train_rmse[k] = []

    val_rmse[k] = []

    sil_score[k] = []

      

    st = time.time()

    for i in range(5):



      sc = StandardScaler()

      train = pd.merge( pd.DataFrame( data = train_id[i].reshape(-1), columns = ['id']), inp )

      

      text = train[train[column].apply(lambda x: len(x) >= 1)][column] 

      item_to_idx, emb_matrix = get_emb_by_pmi(text = text, column = column, top_w = top_w, flag_svd = flag_svd)

      train['encode'] = train[column].apply(lambda x: [0] if len([w for w in x if item_to_idx.get(w) is not None]) == 0 else [item_to_idx.get(w) for w in x if item_to_idx.get(w) is not None] ) 

      train['emb'] = train['encode'].apply(lambda x: np.mean(np.vstack([emb_matrix[i] for i in x]), axis = 0) )

      x_train = np.stack(train['emb'].values)

      x_train_sc = sc.fit_transform(x_train)

      

      val = pd.merge( pd.DataFrame( data = val_id[i].reshape(-1), columns = ['id']), inp )

      

      

      val['encode'] = val[column].apply(lambda x: [0] if len([w for w in x if item_to_idx.get(w) is not None]) == 0 else [item_to_idx.get(w) for w in x if item_to_idx.get(w) is not None])

      val['emb'] = val['encode'].apply(lambda x: np.mean(np.vstack([emb_matrix[i] for i in x]), axis = 0) )

      x_val = np.stack(val['emb'].values)

      x_val_sc = sc.fit_transform(x_val)





      kmeans = KMeans(n_clusters = k, random_state = 42)

      kmeans.fit(x_train_sc)



      train['label'] = kmeans.predict(x_train_sc)

      train_group = train.groupby('label').aggregate({'log_revenue': ['mean','count']}).reset_index()

      train_group.columns = train_group.columns.map('_'.join).str.strip('_')



      train = pd.merge(train , train_group , on = 'label', how ='left')

      val['label'] = kmeans.predict(x_val_sc)

      val = pd.merge(val , train_group , on = 'label', how = 'left')

      val['log_revenue_mean'] = val['log_revenue_mean'].fillna(train['log_revenue'].mean())   



      train_rmse_base = np.sqrt(mean_squared_error(train['log_revenue'], np.repeat(train['log_revenue'].mean() , train.shape[0]))) 



      base_rmse[k].append(train_rmse_base)

      sil_score[k].append(silhouette_score(x_train_sc, kmeans.labels_))

      train_rmse[k].append(np.sqrt(mean_squared_error(train['log_revenue'], train['log_revenue_mean']) ))

      val_rmse[k].append(np.sqrt(mean_squared_error(val['log_revenue'], val['log_revenue_mean']) ))

    print(k, 'clusters Done', (time.time() - st)/60, 'min' )

    

  return  base_rmse, train_rmse,val_rmse, sil_score





def get_pmi_cluster(inp, column, top_w, nums_cluster, item_to_idx, emb_matrix, sc, kmeans,  flag_svd = False, flag_train = True):

  

     

  text = inp[inp[column].apply(lambda x: len(x) >= 1)][column] 



  if flag_train:

    sc = StandardScaler()



    item_to_idx, emb_matrix = get_emb_by_pmi(text = text, column = column, top_w = top_w, flag_svd = flag_svd)



    inp['encode'] = inp[column].apply(lambda x: [0] if len([w for w in x if item_to_idx.get(w) is not None]) == 0 else [item_to_idx.get(w) for w in x if item_to_idx.get(w) is not None] ) 

    inp['emb'] = inp['encode'].apply(lambda x: np.mean(np.vstack([emb_matrix[i] for i in x]), axis = 0) )

    x = np.stack(inp['emb'].values)

    sc.fit(x)

    x_sc = sc.transform(x)

    kmeans = KMeans(n_clusters = nums_cluster, random_state = 42)

    kmeans.fit(x_sc)



  else:



    inp['encode'] = inp[column].apply(lambda x: [0] if len([w for w in x if item_to_idx.get(w) is not None]) == 0 else [item_to_idx.get(w) for w in x if item_to_idx.get(w) is not None] ) 

    inp['emb'] = inp['encode'].apply(lambda x: np.mean(np.vstack([emb_matrix[i] for i in x]), axis = 0) )

    x = np.stack(inp['emb'].values)

    x_sc = sc.transform(x)





  inp['label_' + column] = kmeans.predict(x_sc)



  return  inp, item_to_idx, emb_matrix, sc, kmeans

df = pd.DataFrame(sorted(dict(Counter(input_data['country_list'].sum())).items(), key = lambda x: x[1], reverse = True))

df['rate'] = df[1].cumsum()/df[1].sum()

fig, ax = plt.subplots(1,2, figsize = (12,4))

ax[0].plot(df['rate'])

ax[0].set_title('Cumulative plot per keywords')

ax[1].barh( input_data['country_list'].apply(lambda x: len(x)).value_counts().index, 

         input_data['country_list'].apply(lambda x: len(x)).value_counts() )

ax[1].set_title('Amout of films VS number of production countries')



print(df.head(5))

print('25% of actors cover 50-60%  of all cast')
base_rmse, train_rmse,val_rmse, sil_score = fit_pmi_cluster(inp = input_data[['id','log_revenue','country_list']], column = 'country_list', 

                                                            top_w = 1 , nums_cluster = [2,4,8,16] )
nums_cluster = [2,4,8,16]

fig, ax = plt.subplots(1,1, figsize = (8,6))

ax.plot(nums_cluster,[np.mean(base_rmse[i]) for i in nums_cluster], ls = '--', color = 'black' )

ax.plot(nums_cluster,[np.mean(train_rmse[i]) for i in nums_cluster], color = 'red' )

ax.plot(nums_cluster,[np.mean(val_rmse[i]) for i in nums_cluster] , color = 'blue')

ax.set_title('RMSE VS Number of clusters')

ax.legend(['base rmse','mean train rmse','mean val rmse'])

pprint([(k,np.mean(x)) for k,x in val_rmse.items()])
input_data, country_item_to_idx, country_emb_matrix, country_sc, country_kmeans = get_pmi_cluster(input_data, column = 'country_list', top_w = 1,

                                                           nums_cluster = 4, 

                                                           item_to_idx = None,

                                                           emb_matrix = None,

                                                           sc = None,

                                                           kmeans = None,

                                                           flag_svd = False, 

                                                           flag_train = True)



test, _, _, _, _ = get_pmi_cluster(test, column = 'country_list', top_w = 1,

                                                           nums_cluster = 4, 

                                                           item_to_idx = country_item_to_idx,

                                                           emb_matrix = country_emb_matrix,

                                                           sc = country_sc,

                                                           kmeans = country_kmeans,

                                                           flag_svd = False, 

                                                           flag_train = False)

base_rmse, train_rmse,val_rmse, sil_score = fit_pmi_cluster(inp = input_data[['id','log_revenue','genre_list']], column = 'genre_list', top_w = 1 , nums_cluster = [2,4,8,16,32] )
nums_cluster = [2,4,8,16,32]

fig, ax = plt.subplots(1,2, figsize = (15,6))

ax[0].plot(nums_cluster,[np.mean(base_rmse[i]) for i in nums_cluster], ls = '--', color = 'black' )

ax[0].plot(nums_cluster,[np.mean(train_rmse[i]) for i in nums_cluster], color = 'red' )

ax[0].plot(nums_cluster,[np.mean(val_rmse[i]) for i in nums_cluster] , color = 'blue')

ax[0].set_title('RMSE VS Number of clusters')

ax[0].legend(['base rmse','mean train rmse','mean val rmse'])

ax[1].plot(nums_cluster,[np.mean(sil_score[i]) for i in nums_cluster] )



ax[1].set_title('Silhouette score VS Number of clusters')
input_data, genre_item_to_idx, genre_emb_matrix, genre_sc, genre_kmeans = get_pmi_cluster(input_data, column = 'genre_list', top_w = 1,

                                                           nums_cluster = 8, 

                                                           item_to_idx = None,

                                                           emb_matrix = None,

                                                           sc = None,

                                                           kmeans = None,

                                                           flag_svd = False, 

                                                           flag_train = True)



test, _, _, _, _ = get_pmi_cluster(test, column = 'genre_list', top_w = 1,

                                                           nums_cluster = 8, 

                                                           item_to_idx = genre_item_to_idx,

                                                           emb_matrix = genre_emb_matrix,

                                                           sc = genre_sc,

                                                           kmeans = genre_kmeans,

                                                           flag_svd = False, 

                                                           flag_train = False)
# df = pd.DataFrame(sorted(dict(Counter(keywords.sum())).items(), key = lambda x: x[1], reverse = True))

df = pd.DataFrame(sorted(dict(Counter(input_data['keywords_list'].sum())).items(), key = lambda x: x[1], reverse = True))



df['rate'] = df[1].cumsum()/df[1].sum()

plt.plot(df['rate'])

plt.title('Cumulative plot per keywords')

print(df.head(5))

print('Top base keywords & cummulative plot of all keywords we can get 80% of all by using ~ 3000 ')
base_rmse, train_rmse,val_rmse, sil_score = fit_pmi_cluster(inp = input_data[['id','log_revenue','keywords_list']], column = 'keywords_list', top_w = 0.8 , 

                                                        nums_cluster = [2,4,8,16] )
nums_cluster = [2,4,8,16]

fig, ax = plt.subplots(1,2, figsize = (15,6))

ax[0].plot(nums_cluster,[np.mean(base_rmse[i]) for i in nums_cluster], ls = '--', color = 'black' )

ax[0].plot(nums_cluster,[np.mean(train_rmse[i]) for i in nums_cluster], color = 'red' )

ax[0].plot(nums_cluster,[np.mean(val_rmse[i]) for i in nums_cluster] , color = 'blue')

ax[0].set_title('RMSE VS Number of clusters')

ax[0].legend(['base rmse','mean train rmse','mean val rmse'])

ax[1].plot(nums_cluster,[np.mean(sil_score[i]) for i in nums_cluster] )



ax[1].set_title('Silhouette score VS Number of clusters')
input_data, keyword_item_to_idx, keyword_emb_matrix, keyword_sc, keyword_kmeans = get_pmi_cluster(input_data, column = 'keywords_list', top_w = 1,

                                                           nums_cluster = 8, 

                                                           item_to_idx = None,

                                                           emb_matrix = None,

                                                           sc = None,

                                                           kmeans = None,

                                                           flag_svd = False, 

                                                           flag_train = True)



test, _, _, _, _ = get_pmi_cluster(test, column = 'keywords_list', top_w = 1,

                                                           nums_cluster = 8, 

                                                           item_to_idx = keyword_item_to_idx,

                                                           emb_matrix = keyword_emb_matrix,

                                                           sc = keyword_sc,

                                                           kmeans = keyword_kmeans,

                                                           flag_svd = False, 

                                                           flag_train = False)
inp =  input_data[['id','crew']].dropna().set_index('id')

###



crew_departments  = inp['crew'].apply(lambda x: [item['department'] for item in eval(x) ] )

crew_gender = pd.DataFrame( data = inp['crew'].apply(lambda x: [item['gender'] for item in eval(x) ] ).values,

                           index = inp.index )

crew_dep_dict = dict(Counter(crew_departments.sum()))

###



crew_df = pd.DataFrame( data = np.stack(crew_departments.apply(lambda x: [dict(Counter(x)).get(d,0) for d,_ in crew_dep_dict.items()] ).values),

             columns = ['crew.' + w for w in list(crew_dep_dict.keys())], 

                      index = inp.index )



sum_ = crew_df.sum(axis = 1)

max_ = crew_df.max(axis = 1)

crew_df['sum_crew_dep'] = sum_

crew_df['max_crew_dep'] = max_



crew_gender['male_crew'] = crew_gender[0].apply(lambda x: sum([w == 2 for w in x]))

crew_gender['female_crew'] = crew_gender[0].apply(lambda x: sum([w == 1 for w in x]))

crew_gender['def_gender_crew'] = crew_gender['male_crew'] + crew_gender['female_crew']



###



inp = pd.concat([inp,crew_df, crew_gender[['male_crew','female_crew','def_gender_crew']]], axis = 1)

inp['argmax_crew_dep'] = inp[['crew.' + v for v in list(crew_dep_dict.keys())]].apply(lambda x: list(crew_dep_dict.keys())[np.argmax(np.array(x))], axis = 1)

inp1 = pd.merge(input_data[['id','revenue','release_year']], inp.reset_index().rename(columns = {'index': 'id'}) , how = 'left', on = 'id')



df_part = (inp1['sum_crew_dep'].value_counts().sort_index().cumsum()/inp1['sum_crew_dep'].shape[0]).reset_index()



########

fig, ax = plt.subplots(2,3, figsize = (16,8))

axi = ax.flatten()

axi[0].plot(df_part['index'], df_part['sum_crew_dep'])

axi[0].set_title('Percent of films VS amount of crew on film')

axi[1].bar(inp1['sum_crew_dep'].value_counts().index, inp1['sum_crew_dep'].value_counts().values )

axi[1].set_title('Histogram per amount of crew')

axi[2].bar(list(inp1.groupby('release_year').aggregate({'sum_crew_dep':'mean'}).index), inp1.groupby('release_year').aggregate({'sum_crew_dep':'mean'}).values.reshape(-1) )

axi[2].set_title('Mean amount of crew per release year')





axi[3].bar(list(inp1.groupby('release_year').count().index), inp1.groupby('release_year').aggregate({'def_gender_crew':'sum'}).values.reshape(-1)/inp1.groupby('release_year').aggregate({'sum_crew_dep':'sum'}).values.reshape(-1) )

axi[3].set_ylim([0,1])

axi[3].set_title('Amount of crew with defined gender per release year')



axi[4].bar(list(inp1.groupby('release_year').count().index), inp1.groupby('release_year').aggregate({'male_crew':'sum'}).values.reshape(-1)/inp1.groupby('release_year').aggregate({'sum_crew_dep':'sum'}).values.reshape(-1) )

axi[4].set_ylim([0,1])

axi[4].set_title('Amount of crew with male gender per release year')



axi[5].bar(list(inp1.groupby('release_year').count().index), inp1.groupby('release_year').aggregate({'female_crew':'sum'}).values.reshape(-1)/inp1.groupby('release_year').aggregate({'sum_crew_dep':'sum'}).values.reshape(-1) )

axi[5].set_ylim([0,1])

axi[5].set_title('Amount of crew with female gender per release year')

plt.tight_layout()

plt.show()

df = pd.DataFrame(sorted(dict(Counter(input_data['crew_list'].sum())).items(), key = lambda x: x[1], reverse = True))

df['rate'] = df[1].cumsum()/df[1].sum()

plt.plot(df['rate'])

plt.title('Cumulative plot per crew')

print(df.head(5))

print('25% of crew cover 50-60%  of all crew')
base_rmse, train_rmse,val_rmse, sil_score = fit_pmi_cluster(inp = input_data[['id','log_revenue','crew_list']], column = 'crew_list', top_w = 0.3, 

                                                        nums_cluster = [2,4,8,16], flag_svd = False )
nums_cluster = [2,4,8,16]

fig, ax = plt.subplots(1,2, figsize = (15,6))

ax[0].plot(nums_cluster,[np.mean(base_rmse[i]) for i in nums_cluster], ls = '--', color = 'black' )

ax[0].plot(nums_cluster,[np.mean(train_rmse[i]) for i in nums_cluster], color = 'red' )

ax[0].plot(nums_cluster,[np.mean(val_rmse[i]) for i in nums_cluster] , color = 'blue')

ax[0].set_title('RMSE VS Number of clusters')

ax[0].legend(['base rmse','mean train rmse','mean val rmse'])

ax[1].plot(nums_cluster,[np.mean(sil_score[i]) for i in nums_cluster] )



ax[1].set_title('Silhouette score VS Number of clusters')
input_data, crew_item_to_idx, crew_emb_matrix, crew_sc, crew_kmeans = get_pmi_cluster(input_data, column = 'crew_list', top_w = 0.3,

                                                           nums_cluster = 8, 

                                                           item_to_idx = None,

                                                           emb_matrix = None,

                                                           sc = None,

                                                           kmeans = None,

                                                           flag_svd = False, 

                                                           flag_train = True)





test, _, _, _, _ = get_pmi_cluster(test, column = 'crew_list', top_w = 0.3,

                                                           nums_cluster = 8, 

                                                           item_to_idx = crew_item_to_idx,

                                                           emb_matrix = crew_emb_matrix,

                                                           sc = crew_sc,

                                                           kmeans = crew_kmeans,

                                                           flag_svd = False, 

                                                           flag_train = False)

inp =  input_data[['id','cast']].dropna().set_index('id')

###



cast_gender = pd.DataFrame( data = inp['cast'].apply(lambda x: [item['gender'] for item in eval(x) ] ).values,

                           index = inp.index )





cast_gender['male_cast'] = cast_gender[0].apply(lambda x: sum([w == 2 for w in x]))

cast_gender['female_cast'] = cast_gender[0].apply(lambda x: sum([w == 1 for w in x]))

cast_gender['sum_cast'] = cast_gender[0].apply(lambda x: len(x))

cast_gender['def_gender_cast'] = cast_gender['male_cast'] + cast_gender['female_cast']



###



inp = pd.concat([inp, cast_gender[['male_cast','female_cast','def_gender_cast', 'sum_cast']]], axis = 1)

inp1 = pd.merge(input_data[['id','revenue','release_year']], inp.reset_index().rename(columns = {'index': 'id'}) , how = 'left', on = 'id')



df_part = (inp1['sum_cast'].value_counts().sort_index().cumsum()/inp1['sum_cast'].shape[0]).reset_index()



########

fig, ax = plt.subplots(2,3, figsize = (16,8))

axi = ax.flatten()

axi[0].plot(df_part['index'], df_part['sum_cast'])

axi[0].set_title('Percent of films VS amount of cast on film')

axi[1].bar(inp1['sum_cast'].value_counts().index, inp1['sum_cast'].value_counts().values )

axi[1].set_title('Histogram per amount of cast')

axi[2].bar(list(inp1.groupby('release_year').aggregate({'sum_cast':'mean'}).index), inp1.groupby('release_year').aggregate({'sum_cast':'mean'}).values.reshape(-1) )

axi[2].set_title('Mean amount of cast per release year')





axi[3].bar(list(inp1.groupby('release_year').count().index), inp1.groupby('release_year').aggregate({'def_gender_cast':'sum'}).values.reshape(-1)/inp1.groupby('release_year').aggregate({'sum_cast':'sum'}).values.reshape(-1) )

axi[3].set_ylim([0,1])

axi[3].set_title('Amount of cast with defined gender per release year')



axi[4].bar(list(inp1.groupby('release_year').count().index), inp1.groupby('release_year').aggregate({'male_cast':'sum'}).values.reshape(-1)/inp1.groupby('release_year').aggregate({'sum_cast':'sum'}).values.reshape(-1) )

axi[4].set_ylim([0,1])

axi[4].set_title('Amount of cast with male gender per release year')



axi[5].bar(list(inp1.groupby('release_year').count().index), inp1.groupby('release_year').aggregate({'female_cast':'sum'}).values.reshape(-1)/inp1.groupby('release_year').aggregate({'sum_cast':'sum'}).values.reshape(-1) )

axi[5].set_ylim([0,1])

axi[5].set_title('Amount of cast with female gender per release year')

plt.tight_layout()

plt.show()

df = pd.DataFrame(sorted(dict(Counter(input_data['cast_list'].sum())).items(), key = lambda x: x[1], reverse = True))

df['rate'] = df[1].cumsum()/df[1].sum()

plt.plot(df['rate'])

plt.title('Cumulative plot per cast')

print(df.head(5))

print('25% of cast cover 50-60%  of all cast')
base_rmse, train_rmse,val_rmse, sil_score = fit_pmi_cluster(inp = input_data[['id','log_revenue','cast_list']], column = 'cast_list', top_w = 0.3, 

                                                        nums_cluster = [2,4,8,16], flag_svd = False )
nums_cluster = [2,4,8,16]

fig, ax = plt.subplots(1,2, figsize = (15,6))

ax[0].plot(nums_cluster,[np.mean(base_rmse[i]) for i in nums_cluster], ls = '--', color = 'black' )

ax[0].plot(nums_cluster,[np.mean(train_rmse[i]) for i in nums_cluster], color = 'red' )

ax[0].plot(nums_cluster,[np.mean(val_rmse[i]) for i in nums_cluster] , color = 'blue')

ax[0].set_title('RMSE VS Number of clusters')

ax[0].legend(['base rmse','mean train rmse','mean val rmse'])

ax[1].plot(nums_cluster,[np.mean(sil_score[i]) for i in nums_cluster] )



ax[1].set_title('Silhouette score VS Number of clusters')
input_data, cast_item_to_idx, cast_emb_matrix, cast_sc, cast_kmeans = get_pmi_cluster(input_data, column = 'cast_list', top_w = 0.3,

                                                           nums_cluster = 8, 

                                                           item_to_idx = None,

                                                           emb_matrix = None,

                                                           sc = None,

                                                           kmeans = None,

                                                           flag_svd = False, 

                                                           flag_train = True)





test, _, _, _, _ = get_pmi_cluster(test, column = 'cast_list', top_w = 0.3,

                                                           nums_cluster = 8, 

                                                           item_to_idx = cast_item_to_idx,

                                                           emb_matrix = cast_emb_matrix,

                                                           sc = cast_sc,

                                                           kmeans = cast_kmeans,

                                                           flag_svd = False, 

                                                           flag_train = False)

df = pd.DataFrame(sorted(dict(Counter(input_data['prod_list'].sum())).items(), key = lambda x: x[1], reverse = True))

df['rate'] = df[1].cumsum()/df[1].sum()

plt.plot(df['rate'])

plt.title('Cumulative plot per prod companies')

print(df.head(5))

print('Top prod companies & cummulative plot of all prod companies we can get 80% of all by using ~ 3000 ')

print(df[0].nunique())
base_rmse, train_rmse,val_rmse, sil_score = fit_pmi_cluster(inp = input_data[['id','log_revenue','prod_list']], column = 'prod_list', top_w = 0.4, 

                                                        nums_cluster = [2,4,8,16,32,64] )
nums_cluster = [2,4,8,16,32,64]

fig, ax = plt.subplots(1,2, figsize = (15,6))

ax[0].plot(nums_cluster,[np.mean(base_rmse[i]) for i in nums_cluster], ls = '--', color = 'black' )

ax[0].plot(nums_cluster,[np.mean(train_rmse[i]) for i in nums_cluster], color = 'red' )

ax[0].plot(nums_cluster,[np.mean(val_rmse[i]) for i in nums_cluster] , color = 'blue')

ax[0].set_title('RMSE VS Number of clusters')

ax[0].legend(['base rmse','mean train rmse','mean val rmse'])

ax[1].plot(nums_cluster,[np.mean(sil_score[i]) for i in nums_cluster] )



ax[1].set_title('Silhouette score VS Number of clusters')
input_data, prod_item_to_idx, prod_emb_matrix, prod_sc, prod_kmeans = get_pmi_cluster(input_data, column = 'prod_list', top_w = 0.4,

                                                           nums_cluster = 16, 

                                                           item_to_idx = None,

                                                           emb_matrix = None,

                                                           sc = None,

                                                           kmeans = None,

                                                           flag_svd = False, 

                                                           flag_train = True)



test, _, _, _, _ = get_pmi_cluster(test, column = 'prod_list', top_w = 0.4,

                                                           nums_cluster = 16, 

                                                           item_to_idx = prod_item_to_idx,

                                                           emb_matrix = prod_emb_matrix,

                                                           sc = prod_sc,

                                                           kmeans = prod_kmeans,

                                                           flag_svd = False, 

                                                           flag_train = False)





import gensim

# Load Google's pre-trained Word2Vec model.

wordvec = gensim.models.KeyedVectors.load_word2vec_format('../input/wordvec/googlenews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True) 





import nltk

nltk.download('stopwords')

# NLTK Stop words

from nltk.corpus import stopwords

stop_words = stopwords.words('english')





def preprocessing(inp):

  data = inp.values.tolist()



  # Remove single quotes

  data = [re.sub("\'", "", sent) for sent in data]



#   pprint(data[:1]))

  

  return data



def sent_to_words(sentences):

    for sentence in sentences:

        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



  

# !python3 -m spacy download en  # run in terminal once

def process_words(inp_words, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""

    

    bigram = gensim.models.Phrases(inp_words, min_count=5, threshold=10) # higher threshold fewer phrases.

    bigram_mod = gensim.models.phrases.Phraser(bigram)



    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in inp_words]

    texts = [bigram_mod[doc] for doc in texts]

#     texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

    texts_out = []

    nlp = spacy.load('en', disable=['parser', 'ner'])

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    # remove stopwords once more after lemmatization

    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    

    return texts_out



  

  

def format_topics_sentences(ldamodel, corpus, texts):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row_list in enumerate(ldamodel[corpus]):

        row = row_list[0] if ldamodel.per_word_topics else row_list            

        # print(row)

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)
def average_vec_sentences(df, w2id, emb_matrix, column, wv_model, flag_train = True):

  

  data = preprocessing(df[column].dropna())

  data_words = list(sent_to_words(data))

  data_ready = process_words(inp_words = data_words)

  

  if flag_train:

    id2word = corpora.Dictionary(data_ready)

    w2id = dict([(w,i+1) for i,w in enumerate([w for w in list(id2word.token2id.keys()) if wv_model.vocab.get(w) is not None ] ) ])

    emb_matrix = np.vstack( [ np.mean(np.stack([wv_model.wv[w] for w,_ in w2id.items()]), axis = 0, keepdims = True),

    np.stack([wv_model.wv[w] for w,_ in w2id.items()])

       ]

                          )

    

  data_encode = pd.Series(data_ready).apply(lambda x: [w2id.get(w) for w in x if w2id.get(w) is not None] if len([w for w in x if w2id.get(w) is not None]) > 0 else [0])

  X = np.stack(data_encode.apply(lambda x: np.mean([emb_matrix[i,:] for i in x], axis = 0)).values)

  return X, w2id, emb_matrix





def fit_av_cluster(inp, column, wv_model, nums_cluster = [2,4,8,16,32,64]):

  

  base_rmse = {}

  train_rmse = {}

  val_rmse = {}

  sil_score = {}

    

  for k in nums_cluster :

    

    base_rmse[k] = []

    train_rmse[k] = []

    val_rmse[k] = []

    sil_score[k] = []

      

    st = time.time()

    

    for i in range(5):



      train = pd.merge( pd.DataFrame( data = train_id[i].reshape(-1), columns = ['id']), inp )



      x_train, w2id, emb_matrix = average_vec_sentences(train[['id',column]].dropna(), w2id = None, emb_matrix=None,column = column, wv_model = wv_model, flag_train = True)

      kmeans = KMeans(n_clusters = k, random_state = 42)

      kmeans.fit(x_train)



      train = pd.merge( train[['id',column,'log_revenue']], 

      pd.DataFrame( {'id' : train[~train[column].isna()]['id'], 'label' : kmeans.predict(x_train)} ).reset_index().rename(columns= {'index': 'Document_No'} ) , how = 'left')

      

      

      train['label'].fillna(-1, inplace = True)

      

      train_group = train.groupby('label').aggregate({'log_revenue': ['mean','count']}).reset_index()

      train_group.columns = train_group.columns.map('_'.join).str.strip('_')



      train = pd.merge(train , train_group , on = 'label', how ='left')

      

      

      

      val = pd.merge( pd.DataFrame( data = val_id[i].reshape(-1), columns = ['id']), inp )



      x_val, _, _ = average_vec_sentences(val[['id',column]].dropna(), w2id = w2id, emb_matrix=emb_matrix,column = column, wv_model=wv_model, flag_train = False)



      val = pd.merge( val[['id',column,'log_revenue']], 

      pd.DataFrame( {'id' : val[~val[column].isna()]['id'], 'label' : kmeans.predict(x_val)} ).reset_index().rename(columns= {'index': 'Document_No'} ) , how = 'left')



      

      val['label'].fillna(-1, inplace = True)

      val = pd.merge(val , train_group , on = 'label', how = 'left')

      

      

      val['log_revenue_mean'] = val['log_revenue_mean'].fillna(train['log_revenue'].mean())   

      train_rmse_base = np.sqrt(mean_squared_error(train['log_revenue'], np.repeat(train['log_revenue'].mean() , train.shape[0]))) 

      

      base_rmse[k].append(train_rmse_base) 

      sil_score[k].append(silhouette_score(x_train, kmeans.labels_))



      train_rmse[k].append(np.sqrt(mean_squared_error(train['log_revenue'], train['log_revenue_mean']) ))

      val_rmse[k].append(np.sqrt(mean_squared_error(val['log_revenue'], val['log_revenue_mean']) ))

    print(k, 'clusters Done', (time.time() - st)/60, 'min' )

    

  return  base_rmse, train_rmse,val_rmse, sil_score







def get_av_cluster(inp, column, wv_model,  nums_cluster, w2id, emb_matrix, kmeans, flag_train = True):

  



  if flag_train:



    x, w2id, emb_matrix = average_vec_sentences(inp[['id',column]].dropna(), w2id = None, emb_matrix=None,column = column, wv_model = wv_model, flag_train = True)

    kmeans = KMeans(n_clusters = nums_cluster, random_state = 42)

    kmeans.fit(x)

      



  else:

    x, _, _ = average_vec_sentences(inp[['id',column]].dropna(), w2id = w2id, emb_matrix=emb_matrix,column = column, wv_model = wv_model, flag_train = False)





  inp = pd.merge( inp, 

  pd.DataFrame( {'id' : inp[~inp[column].isna()]['id'], 'label_' + column : kmeans.predict(x)} ).reset_index().rename(columns= {'index': 'Document_No'} ) , how = 'left')





  inp['label_' + column].fillna(-1, inplace = True)



  return  inp, w2id, emb_matrix, kmeans

base_rmse, train_rmse,val_rmse, sil_score = fit_av_cluster(input_data[['id','log_revenue','tagline']], column = 'tagline', wv_model = wordvec, nums_cluster = [2,4,8])
nums_cluster = [2,4,8]

fig, ax = plt.subplots(1,1, figsize = (8,6))

ax.plot(nums_cluster,[np.mean(base_rmse[i]) for i in nums_cluster], ls = '--', color = 'black' )

ax.plot(nums_cluster,[np.mean(train_rmse[i]) for i in nums_cluster], color = 'red' )

ax.plot(nums_cluster,[np.mean(val_rmse[i]) for i in nums_cluster] , color = 'blue')

ax.set_title('RMSE VS Number of clusters')

ax.legend(['base rmse','mean train rmse','mean val rmse'])

pprint([(k,np.mean(x)) for k,x in val_rmse.items()])
input_data, tag_w2id, tag_emb_matrix, tag_kmeans = get_av_cluster(input_data, 

                                                                           column = 'tagline',

                                                                                  wv_model = wordvec, 

                                                           nums_cluster = 2, 

                                                           w2id = None,

                                                           emb_matrix = None,

                                                          

                                                           kmeans = None,

                                                     

                                                           flag_train = True)







test, _, _, _  = get_av_cluster(test, column = 'tagline',wv_model = wordvec,

                                                           nums_cluster = 2, 

                                                           w2id = tag_w2id,

                                                           emb_matrix = tag_emb_matrix,

                                                         

                                                           kmeans = tag_kmeans,

                                                        

                                                           flag_train = False)
plt.hist(input_data['overview'].fillna('').apply(lambda x: len(x.split(' '))))

plt.title('word length of overview')

plt.show()
base_rmse, train_rmse,val_rmse, sil_score = fit_av_cluster(input_data[['id','log_revenue','overview_fst']], column = 'overview_fst', wv_model = wordvec, nums_cluster = [2,4,8,16])

nums_cluster = [2,4,8,16]

fig, ax = plt.subplots(1,1, figsize = (8,6))

ax.plot(nums_cluster,[np.mean(base_rmse[i]) for i in nums_cluster], ls = '--', color = 'black' )

ax.plot(nums_cluster,[np.mean(train_rmse[i]) for i in nums_cluster], color = 'red' )

ax.plot(nums_cluster,[np.mean(val_rmse[i]) for i in nums_cluster] , color = 'blue')

ax.set_title('RMSE VS Number of clusters')

ax.legend(['base rmse','mean train rmse','mean val rmse'])

pprint([(k,np.mean(x)) for k,x in val_rmse.items()])
input_data, overview_w2id, overview_emb_matrix, overview_kmeans = get_av_cluster(input_data, 

                                                                           column = 'overview_fst',

                                                                                  wv_model = wordvec, 

                                                           nums_cluster = 4, 

                                                           w2id = None,

                                                           emb_matrix = None,

                                                          

                                                           kmeans = None,

                                                     

                                                           flag_train = True)



test, _, _, _  = get_av_cluster(test, column = 'overview_fst',wv_model = wordvec,

                                                           nums_cluster = 4, 

                                                           w2id = tag_w2id,

                                                           emb_matrix = tag_emb_matrix,

                                                         

                                                           kmeans = tag_kmeans,

                                                        

                                                           flag_train = False)
def fit_img_cluster(inp, input_data_img, nums_cluster = [2,4,8,16,32,64,128]):

  

  base_rmse = {}

  train_rmse = {}

  val_rmse = {}

  sil_score = {}

    

  for k in nums_cluster :

    

    base_rmse[k] = []

    train_rmse[k] = []

    val_rmse[k] = []

    sil_score[k] = []

      

    st = time.time()

    for i in range(5):



      sc = StandardScaler()

      

      

      train = pd.merge( pd.DataFrame( data = train_id[i].reshape(-1), columns = ['id']), input_data_img )

      x_train = train.drop(columns = ['id']).values

      x_train_sc = sc.fit_transform(x_train)



      val = pd.merge( pd.DataFrame( data = val_id[i].reshape(-1), columns = ['id']), input_data_img )

      x_val = val.drop(columns = ['id']).values

      x_val_sc = sc.transform(x_val)



      kmeans = KMeans(n_clusters = k, random_state = 42)

      kmeans.fit(x_train_sc)



      train['label'] = kmeans.predict(x_train_sc)

      train = pd.merge(  pd.DataFrame( data = train_id[i].reshape(-1), columns = ['id']) , train, how = 'left' )

      train = pd.merge( train, input_data[['id','revenue']] )

      train['label'].fillna(0, inplace = True)

      

      

      train_group = train.groupby('label').aggregate({'revenue': ['mean','count']}).reset_index()

      train_group.columns = train_group.columns.map('_'.join).str.strip('_')

     

      train = pd.merge(train , train_group , on = 'label', how ='left')

      

      val['label'] = kmeans.predict(x_val_sc)

      val = pd.merge(  pd.DataFrame( data = val_id[i].reshape(-1), columns = ['id']) , val, how = 'left' )

      val = pd.merge( val, input_data[['id','revenue']])

      

      val['label'].fillna(0, inplace = True)

      

      val = pd.merge(val , train_group , on = 'label', how ='left')

      val['revenue_mean'] = val['revenue_mean'].fillna(train['revenue'].mean())   

      

      train_rmse_base = np.sqrt(mean_squared_error(train['revenue'], np.repeat(train['revenue'].mean() , train.shape[0]))) 



      base_rmse[k].append(train_rmse_base)

      sil_score[k].append(silhouette_score(x_train_sc, kmeans.labels_))

      train_rmse[k].append(np.sqrt(mean_squared_error(train['revenue'], train['revenue_mean']) ))

      val_rmse[k].append(np.sqrt(mean_squared_error(val['revenue'], val['revenue_mean']) ))

    print(k, 'clusters Done', (time.time() - st)/60, 'min' )

    

  return  base_rmse, train_rmse,val_rmse, sil_score





def get_img_cluster(inp, input_data_img,  nums_cluster, sc, kmeans, flag_train = True):

  



  if flag_train:

      

      sc = StandardScaler()

        

      data = pd.merge( inp[['id']], input_data_img )

      x = data.drop(columns = ['id']).values

      x_sc = sc.fit_transform(x)



      kmeans = KMeans(n_clusters = nums_cluster, random_state = 42)

      kmeans.fit(x_sc)



      data['img_label'] = kmeans.predict(x_sc)

      inp = pd.merge( inp , data[['id','img_label']], how = 'left' )

      inp['img_label'].fillna(0, inplace = True)

      



  else:

    data = pd.merge( inp[['id']], input_data_img )

    x = data.drop(columns = ['id']).values

    x_sc = sc.transform(x)



    data['img_label' ] = kmeans.predict(x_sc)

    inp = pd.merge( inp , data[['id','img_label']], how = 'left' )

    inp['img_label'].fillna(0, inplace = True)

    



  return  inp, sc, kmeans
model_resnet50 = ResNet50(weights='imagenet', include_top=False, pooling = 'avg')

# model_resnet50.summary()



img_path = '../input/film_posters/posters-20190619t213659z-001/posters/train/1.jpg'

img = image.load_img(img_path, target_size=(224, 224))

img_data = image.img_to_array(img)

img_data = np.expand_dims(img_data, axis=0)

img_data = resnet50_preprocess(img_data)



resnet50_feature = model_resnet50.predict(img_data)



print (resnet50_feature.shape)
img_path = '../input/film_posters/posters-20190619t213659z-001/posters/train/1.jpg'

image.load_img(img_path, target_size=(224, 224))
resnet50_feature_list = []

img_id = []



for idx, f in tqdm(enumerate(os.listdir('../input/film_posters/posters-20190619t213659z-001/posters/train/'))):

#     print(f)

    f = f.split('.')[0]

    try:

      img_path = '../input/film_posters/posters-20190619t213659z-001/posters/train/' + f + '.jpg'

      img = image.load_img(img_path, target_size=(224, 224))

      img_data = image.img_to_array(img)

      img_data = np.expand_dims(img_data, axis=0)

      img_data = resnet50_preprocess(img_data)

      

      resnet50_feature = model_resnet50.predict(img_data)

      resnet50_feature_np = np.array(resnet50_feature)

      resnet50_feature_list.append(resnet50_feature_np.flatten())

      img_id.append(f)

    except:

      continue 

        

resnet50_feature_list_np = np.array(resnet50_feature_list)



img_df = pd.concat( [ pd.DataFrame({'id': img_id}), pd.DataFrame(resnet50_feature_list_np, columns = ['f_' + str(i) for i in range(resnet50_feature_list_np.shape[1])]) ] , axis = 1 )

img_df['id'] = img_df['id'].astype('int32')

input_data_img = pd.merge(input_data[['id']], img_df , how = 'inner')
nums_cluster = [2,4,8]

base_rmse, train_rmse,val_rmse, sil_score = fit_img_cluster(input_data, input_data_img, nums_cluster = [2,4,8])
fig, ax = plt.subplots(1,2, figsize = (15,6))

ax[0].plot(nums_cluster,[np.mean(base_rmse[i]) for i in nums_cluster], ls = '--', color = 'black' )

ax[0].plot(nums_cluster,[np.mean(train_rmse[i]) for i in nums_cluster], color = 'red' )

ax[0].plot(nums_cluster,[np.mean(val_rmse[i]) for i in nums_cluster] , color = 'blue')

ax[0].set_title('RMSE VS Number of clusters')

ax[0].legend(['base rmse','mean train rmse','mean val rmse'])

ax[1].plot(nums_cluster,[np.mean(sil_score[i]) for i in nums_cluster] )



ax[1].set_title('Silhouette score VS Number of clusters')
sc = StandardScaler()

train = pd.merge( pd.DataFrame( data = train_id[0].reshape(-1), columns = ['id']), input_data_img )

x_train = train.drop(columns = ['id']).values

x_train_sc = sc.fit_transform(x_train)



kmeans = KMeans(n_clusters =4, random_state = 42)

kmeans.fit(x_train_sc)

train['label'] = kmeans.predict(x_train_sc)

print(train.groupby('label').aggregate({'id':'count'}).rename(columns = {'id':'count'}).T)

fig, ax = plt.subplots(3,4, figsize = (15,10))



for i in range(3):

  for j in range(4):

    img_path = '../input/film_posters/posters-20190619t213659z-001/posters/train/' + str(train[train['label'] == j]['id'].sample(1).iloc[0]) + '.jpg'       

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    ax[i][j].imshow(img)

    ax[i][j].set_title('label ' + str(j))

    ax[i][j].axis('off')

plt.show()
input_data, img_sc, img_kmeans = get_img_cluster(input_data, 

                                                 input_data_img,  

                                                 nums_cluster = 4 , 

                                                 sc = None, 

                                                 kmeans = None,

                                                 flag_train = True)
resnet50_feature_list = []

img_id = []



for idx, f in tqdm(enumerate(os.listdir('../input/film_posters/posters-20190619t213659z-001/posters/test/'))):

#     print(f)

    f = f.split('.')[0]

    try:

      img_path = '../input/film_posters/posters-20190619t213659z-001/posters/test/' + f + '.jpg'

      img = image.load_img(img_path, target_size=(224, 224))

      img_data = image.img_to_array(img)

      img_data = np.expand_dims(img_data, axis=0)

      img_data = resnet50_preprocess(img_data)

      

      resnet50_feature = model_resnet50.predict(img_data)

      resnet50_feature_np = np.array(resnet50_feature)

      resnet50_feature_list.append(resnet50_feature_np.flatten())

      img_id.append(f)

    except:

      continue 

        

resnet50_feature_list_np = np.array(resnet50_feature_list)



img_df = pd.concat( [ pd.DataFrame({'id': img_id}), pd.DataFrame(resnet50_feature_list_np, columns = ['f_' + str(i) for i in range(resnet50_feature_list_np.shape[1])]) ] , axis = 1 )

img_df['id'] = img_df['id'].astype('int32')

test_img = pd.merge(test[['id']], img_df , how = 'inner')
test,_, _ = get_img_cluster(test, 

                                                 test_img,  

                                                 nums_cluster = 4 , 

                                                 sc = img_sc, 

                                                 kmeans = img_kmeans,

                                                 flag_train = False)
input_data['crew_director'] = input_data['crew'].fillna('').apply(lambda x : [item['name'] for item in eval(x) if  item['job'] == 'Director'  ][0] if  x != '' and len([item['name'] for item in eval(x) if  item['job'] == 'Director'  ]) > 0 else '' )

test['crew_director'] = test['crew'].fillna('').apply(lambda x : [item['name'] for item in eval(x) if  item['job'] == 'Director'  ][0] if  x != '' and len([item['name'] for item in eval(x) if  item['job'] == 'Director'  ]) > 0  else '' )

print('Top crew directors by amount of films')

full_data['crew_director'].value_counts().head(10)
prev_input = pd.merge(input_data[['id','crew_director','release_dt']], full_data[['crew_director','release_dt','log_revenue','id']].rename(columns = {'log_revenue': 'log_revenue_dir',

                                                                                          'release_dt': 'prev_release_dt',

                                                                                                            'id': 'prev_id'}) , on = 'crew_director' )



prev_input = prev_input[(prev_input['prev_release_dt'] < prev_input['release_dt'])& (prev_input['crew_director'] != '')]



prev_input['year_to_lst_film_dir'] = prev_input.apply(lambda x: (x['release_dt']- x['prev_release_dt']).days/365, axis = 1)

prev_group = prev_input.groupby('id').aggregate({'prev_id':'count','year_to_lst_film_dir': 'min', 'log_revenue_dir' : 'mean'}).reset_index().rename(columns = {'prev_id': 'film_count_dir'})



prev_test = pd.merge(test[['id','crew_director','release_dt']] , full_data[['crew_director','release_dt','id', 'log_revenue']].rename(columns = {  'log_revenue': 'log_revenue_dir',

                                                                                          'release_dt': 'prev_release_dt',

                                                                                                            'id': 'prev_id'}) , on = 'crew_director' )



prev_test = prev_test[(prev_test['prev_release_dt'] < prev_test['release_dt'])& (prev_test['crew_director'] != '')]



prev_test['year_to_lst_film_dir'] = prev_test.apply(lambda x: (x['release_dt']- x['prev_release_dt']).days/365, axis = 1)

prev_test_group = prev_test.groupby('id').aggregate({'prev_id':'count','year_to_lst_film_dir': 'min', 'log_revenue_dir' : 'mean'}).reset_index().rename(columns = {'prev_id': 'film_count_dir'})



try:

  test.drop(columns = ['film_count_dir'	, 'year_to_lst_film_dir', 'log_revenue_dir'], inplace = True)

except:

  print('clear')

  

try:

  input_data.drop(columns = ['film_count_dir'	, 'year_to_lst_film_dir', 'log_revenue_dir'], inplace = True)

except:

  print('clear')



input_data = pd.merge(input_data, prev_group, on ='id', how = 'left' )

input_data['year_to_lst_film_dir'].fillna(-100, inplace = True) 

input_data['film_count_dir'].fillna(0, inplace = True)

input_data['log_revenue_dir'].fillna(0, inplace = True)





test = pd.merge(test, prev_test_group, on ='id', how = 'left' )

test['year_to_lst_film_dir'].fillna(-100, inplace = True) 

test['film_count_dir'].fillna(0, inplace = True)

test['log_revenue_dir'].fillna(0, inplace = True)

print( 'train rate without additional info', input_data[input_data['film_count_dir'] == 0].shape[0]/input_data.shape[0] )

print( 'test rate without additional info', test[test['film_count_dir'] == 0].shape[0]/test.shape[0] )
input_data['collection'] = input_data['belongs_to_collection'].fillna('').apply(lambda x : [item['name'] for item in eval(x)][0] if  x != '' else '' )

test['collection'] = test['belongs_to_collection'].fillna('').apply(lambda x : [item['name'] for item in eval(x)][0] if  x != '' else '' )

full_data['collection'] = full_data['belongs_to_collection'].fillna('').apply(lambda x : [item['name'] for item in eval(x)][0] if  x != '' else '' )



prev_collection =  pd.merge(input_data[['id','collection','release_dt']], full_data[['collection','release_dt','id', 'log_revenue']].rename(columns = { 'log_revenue': 'log_revenue_col',

                                                                                          'release_dt': 'prev_col_release_dt',

                                                                                                            'id': 'prev_id'}) , on = 'collection' )



prev_collection = prev_collection[(prev_collection['prev_col_release_dt'] < prev_collection['release_dt'])& (prev_collection['collection'] != '')]



prev_collection['year_to_lst_film_col'] = prev_collection.apply(lambda x: (x['release_dt']- x['prev_col_release_dt']).days/365, axis = 1)

prev_collection = prev_collection.groupby('id').aggregate({'prev_id':'count','year_to_lst_film_col': 'min', 'log_revenue_col': 'mean' }).reset_index().rename(columns = {'prev_id': 'film_count_col'})





prev_test_collection =  pd.merge(test[['id','collection','release_dt']], full_data[['collection','release_dt','id','log_revenue']].rename(columns = { 'log_revenue': 'log_revenue_col',

                                                                                          'release_dt': 'prev_col_release_dt',

                                                                                                            'id': 'prev_id'}) , on = 'collection' )



prev_test_collection = prev_test_collection[(prev_test_collection['prev_col_release_dt'] < prev_test_collection['release_dt'])& (prev_test_collection['collection'] != '')]



prev_test_collection['year_to_lst_film_col'] = prev_test_collection.apply(lambda x: (x['release_dt']- x['prev_col_release_dt']).days/365, axis = 1)

prev_test_collection = prev_test_collection.groupby('id').aggregate({'prev_id':'count','year_to_lst_film_col': 'min',  'log_revenue_col': 'mean'}).reset_index().rename(columns = {'prev_id': 'film_count_col'})

try:

  test.drop(columns = ['film_count_col'	, 'year_to_lst_film_col', ' log_revenue_col'], inplace = True)

except:

  print('clear')

  

try:

  input_data.drop(columns = ['film_count_col'	, 'year_to_lst_film_col',  ' log_revenue_col'], inplace = True)

except:

  print('clear')



input_data = pd.merge(input_data, prev_collection, on ='id', how = 'left' )

input_data['year_to_lst_film_col'].fillna(-100, inplace = True) 

input_data['film_count_col'].fillna(0, inplace = True)

input_data['log_revenue_col'].fillna(0, inplace = True)

 

test = pd.merge(test, prev_test_collection, on ='id', how = 'left' )

test['year_to_lst_film_col'].fillna(-100, inplace = True) 

test['film_count_col'].fillna(0, inplace = True)

test['log_revenue_col'].fillna(0, inplace = True)
print( 'rate without additional info', input_data[input_data['film_count_col'] == 0].shape[0]/input_data.shape[0] )
import seaborn as sns

fig, ax = plt.subplots(1,2, figsize = (25,6)) 

ax[0].bar(input_data['release_year'].value_counts().index, input_data['release_year'].value_counts().values)

ax[0].set_title('Amount of films by release year')



# sns.heatmap(pd.crosstab(input_data['release_month'], input_data['release_day']) , ax=ax[1])

sns.heatmap(pd.crosstab(index = input_data['release_month'], columns = input_data['release_day'],values = input_data['revenue'], aggfunc = 'count') , ax=ax[1])





ax[1].set_title('heatmap : amount of released films per month vs days')

plt.show()
input_data['log_budget'] = input_data['budget'].fillna(0).apply(lambda x: np.log(x+1))

test['log_budget'] = test['budget'].fillna(0).apply(lambda x: np.log(x+1))
#Delete outliers 

# ind = (input_data['log_revenue'] >= (Q1 - 1.5 * IQR)) & (input_data['log_revenue'] <= (Q3 + 1.5 * IQR) ) 

ind = np.repeat(True, input_data.shape[0])

from pygam import LinearGAM, s, l, f, te

gam = LinearGAM(f(0) + f(1) + f(2) + f(3) + f(4) + f(5) + f(6) + f(7) + f(8) +

               s(9) + s(10)  + s(11) + 

               s(12) + s(13) + s(14) +

               s(15) + s(16) +s(17) +s(18) + s(19) +s(20))

gam.gridsearch(input_data[ind][['label_country_list', 'label_genre_list','label_keywords_list', 

                           'label_crew_list', 'label_cast_list', 'label_prod_list', 

                          'label_tagline', 'label_overview_fst', 'img_label',

                            'year_to_lst_film_col','film_count_col', 'log_revenue_col',

          'year_to_lst_film_dir','film_count_dir', 'log_revenue_dir',

                           

                           'popularity','runtime', 'release_year', 'release_month', 'log_budget', 'release_day'

                 

                          ]].values, input_data[ind]['log_revenue'].values)





## plotting

plt.figure();

fig, axs = plt.subplots(4,5 , figsize = (20,15))



axi = axs.flatten()

titles = ['label_country_list', 'label_genre_list', 'label_keywords_list', 

          'label_crew_list', 'label_cast_list', 'label_prod_list',

         'label_tagline', 'label_overview_fst', 'img_label',

          'year_to_lst_film_col','film_count_col', 'log_revenue_col',

          'year_to_lst_film_dir','film_count_dir','log_revenue_dir',

          

          'popularity','runtime', 'release_year', 'release_month', 'log_budget', 'release_day'

         

         

         ]

for i, ax in enumerate(axi):

#     if i > 17:

#       break

    XX = gam.generate_X_grid(term=i)

    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))

    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

    

    

    ax.set_title(titles[i])

best_lam = gam.lam

best_gam = LinearGAM(f(0) + f(1) + f(2) + f(3) + f(4) + f(5) + f(6) + f(7) + f(8) +

               s(9) + s(10)  + s(11) + 

               s(12) + s(13) + s(14) +

               s(15) + s(16) + s(17) + s(18) + s(19) + s(20), lam = best_lam  )

best_gam.fit(input_data[ind][['label_country_list', 'label_genre_list','label_keywords_list', 

                           'label_crew_list', 'label_cast_list', 'label_prod_list', 

                          'label_tagline', 'label_overview_fst', 'img_label',

                            'year_to_lst_film_col','film_count_col', 'log_revenue_col',

          'year_to_lst_film_dir','film_count_dir', 'log_revenue_dir',

                           

                           'popularity','runtime', 'release_year', 'release_month', 'log_budget', 'release_day'

                 

                          ]].values, input_data[ind]['log_revenue'].values)
np.sqrt(mean_squared_error(best_gam.predict(input_data[['label_country_list', 'label_genre_list','label_keywords_list', 

                           'label_crew_list', 'label_cast_list', 'label_prod_list', 

                          'label_tagline', 'label_overview_fst', 'img_label',

                            'year_to_lst_film_col','film_count_col', 'log_revenue_col',

          'year_to_lst_film_dir','film_count_dir', 'log_revenue_dir',

                           

                           'popularity','runtime', 'release_year', 'release_month',  'log_budget', 'release_day'

                 

                          ]].values), 

                           input_data['log_revenue'].values) )
input_data[(input_data['log_revenue'] >= (Q1 - 1.5 * IQR)) & (input_data['log_revenue'] <= (Q3 + 1.5 * IQR) ) ].shape[0]
test_pred = best_gam.predict(test[['label_country_list', 'label_genre_list','label_keywords_list', 

                           'label_crew_list', 'label_cast_list', 'label_prod_list', 

                          'label_tagline', 'label_overview_fst', 'img_label',

                            'year_to_lst_film_col','film_count_col', 'log_revenue_col',

          'year_to_lst_film_dir','film_count_dir', 'log_revenue_dir',

                           

                           'popularity','runtime', 'release_year', 'release_month', 'log_budget' , 'release_day'

                                                  ]].values)
pd.DataFrame( data = {'id' : test['id'], 'revenue': np.exp(test_pred) }).to_csv('gam_submit_model.csv', index = False)