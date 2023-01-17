import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from xgboost import XGBRegressor

from xgboost import plot_importance

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



from keras.models import Model

from keras.layers import Input, Embedding, concatenate, Dense, Dropout, BatchNormalization,Flatten

from keras.regularizers import l2

import keras.backend as K

from keras.models import model_from_json

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

import pickle

from itertools import product



def save_model_xgb(model,filename):

    # this is a helper function used to save xgb model

    #if not os.path.isdir('cache'):

        #os.mkdir('cache')

    pickle.dump(model, open("chache/"+filename+'.pickle', "wb"))               

def read_model_xgb(filename):

    # this is a helper function used to restore xgb model

    return pickle.load(open("../input/models/chache/"+filename+'.pickle',"rb"))

def save_model(model,filename):

    # this is a helper function used to save a keras NN model architecture and weights

    json_string = model.to_json()

    #if not os.path.isdir('cache'):

        #os.mkdir('cache')

    open(os.path.join

         ('../input/models/chache/', filename+'_architecture.json'), 'w').write(json_string)

    model.save_weights(os.path.join('../input/models/chache/', filename+'_model_weights.h5'), overwrite=True)

def read_model(filename):

    # this is a helper function used to restore a keras NN model architecture and weights

    model = model_from_json(open(os.path.join('../input/models/chache/', filename+'_architecture.json')).read())

    model.load_weights(os.path.join('../input/models/chache/', filename+'_model_weights.h5'))

    return model



import os
data = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').drop('ID',axis = 1)
# Aggregate monthly

data = data.groupby(['date_block_num','shop_id','item_id'],axis=0).agg({'item_cnt_day':'sum','item_price':'sum'}).reset_index()

data.columns = ['date_block_num','shop_id','item_id','item_shop_cnt','item_shop_rev']
# The test set contains an entry for all item-shop combinations for a group of stores and items.

# On the other hand our training set does not, so we expand it to better reflect the test set.  

expanded = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = data[data.date_block_num==i]

    expanded.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique()))))    

expanded = pd.DataFrame(np.vstack(expanded), columns=cols)

data = pd.merge(expanded, data, on=cols, how='left')
# Add test month

test['date_block_num'] = 34

data = pd.concat([data, test], ignore_index=True, sort=False, keys=['date_block_num','shop_id','item_id'])
# Clip monthly sales count to (0,20) to reflect test metric and fill na with zeros.                             

data = data.fillna(0)

data['item_shop_cnt'] = data['item_shop_cnt'].clip(0,20)                                
data['cnt'] = data['item_shop_cnt']



# Lag sales features to previous month

shifted_month = data[['date_block_num','shop_id','item_id','item_shop_cnt','item_shop_rev']].copy()

shifted_month['date_block_num'] += 1



data = data.drop(['item_shop_cnt','item_shop_rev'],axis=1)

data = pd.merge(data, shifted_month, on=['date_block_num','shop_id','item_id'], how='left').fillna(0)

data = data.rename(columns={'item_shop_cnt':'item_shop_cnt_prev','item_rev':'item_rev_prev'})
# Add item category feature

data = pd.merge(data,items[['item_id','item_category_id']],on =['item_id'])
# Add month feature (0-11)

data['month'] = data['date_block_num']%12
# Add days in month feature

days = pd.Series([2,0,2,1,2,1,2,2,1,2,1,2])

data['days'] = data['month'].map(days)
# Add item all-shop count

item_allshop_cnt_prev = data.groupby(['date_block_num','item_id'],axis=0).agg({'item_shop_cnt_prev':'sum'}).reset_index()

item_allshop_cnt_prev = item_allshop_cnt_prev.rename(columns={'item_shop_cnt_prev':'item_allshop_cnt_prev'})

data = pd.merge(data,item_allshop_cnt_prev)
# Add all count

all_cnt_prev = data.groupby(['date_block_num'],axis=0).agg({'item_shop_cnt_prev':'sum'}).reset_index()

all_cnt_prev = all_cnt_prev.rename(columns={'item_shop_cnt_prev':'all_cnt_prev'})

data = pd.merge(data,all_cnt_prev)
# Add shop all-item cnt

shop_allitem_cnt_prev = data.groupby(['date_block_num','shop_id'],axis=0).agg({'item_shop_cnt_prev':'sum'}).reset_index()

shop_allitem_cnt_prev = shop_allitem_cnt_prev.rename(columns={'item_shop_cnt_prev':'shop_allitem_cnt_prev'})

data = pd.merge(data,shop_allitem_cnt_prev)
data  = data.fillna(0)

data = data.astype({'date_block_num':np.int8,

             'shop_id':np.int8,

             'item_id':np.int16,

             'cnt':np.int32,

             'item_shop_cnt_prev':np.int32,

             'item_shop_rev':np.float64,

             'item_category_id':np.int8,

             'month':np.int8,

             'days':np.int8,

             'item_allshop_cnt_prev':np.int32,

             'all_cnt_prev':np.int32,

             'shop_allitem_cnt_prev':np.int32,

             })
data.to_pickle('data3.pkl')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').drop('ID',axis = 1)
data = pd.read_pickle('data3.pkl')

x_train = data[data.date_block_num < 33].drop(['cnt'], axis='columns')

y_train = data[data.date_block_num < 33]['cnt']

x_valid = data[data.date_block_num == 33].drop(['cnt'], axis='columns')

y_valid = data[data.date_block_num == 33]['cnt']

x_test = data[data.date_block_num == 34].drop(['cnt'], axis=1)



x_test = pd.merge(test,x_test,on=['item_id','shop_id'], right_index=True).sort_index() # This alignes x_test rows with test rows (for submission) 

x_test = x_test[x_train.columns]
xgb_model= XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)
xgb_model= read_model_xgb('xgb_model')

y_test = xgb_model.predict(x_test).clip(0, 20)

submission = pd.DataFrame({

    "ID": x_test.index, 

    "item_cnt_month": y_test

})

submission.to_csv('xgb_submission.csv', index=False)
data = pd.read_pickle('data3.pkl')

cat_feat = ['shop_id','item_id','item_category_id','month','days'] # Categorical features

noncat_feat = list(set(data.columns)-set(cat_feat)-set(['cnt'])) # Non-categorical features



data = data[['date_block_num']+cat_feat+['cnt']]



x_train = data[data.date_block_num < 33][data.date_block_num != 10].drop(['cnt','date_block_num'], axis='columns')

y_train = data[data.date_block_num < 33][data.date_block_num != 10]['cnt']



x_valid = data[(data.date_block_num == 33) | (data.date_block_num == 10)].drop(['cnt','date_block_num'], axis='columns')

y_valid = data[(data.date_block_num == 33) | (data.date_block_num == 10)]['cnt']



x_test = data[data.date_block_num == 34].drop(['cnt','date_block_num'], axis='columns')



x_test = pd.merge(test,x_test,on=['item_id','shop_id'], right_index=True).sort_index() # This alignes x_test rows with test rows (for submission) 

x_test = x_test[x_train.columns]



x_train_cat = [x_train[feat] for feat in cat_feat]

x_valid_cat = [x_valid[feat] for feat in cat_feat]

x_test_cat =  [x_test[feat] for feat in cat_feat]

embedded = pd.DataFrame({'n':[60,22170,84,12,3],'m':[7,15,9,4,1]}, index = ['shop_id','item_id','item_category_id','month','days'])

embedded
shop_id_inp = Input(shape=(1,),dtype='int8')

item_id_inp = Input(shape=(1,),dtype='int16')

item_category_id_inp = Input(shape=(1,),dtype='int8')

month_inp = Input(shape=(1,),dtype='int8')

days_inp = Input(shape=(1,),dtype='int8')



shop_id_emb = Embedding(60,7,input_length=1, embeddings_regularizer=l2(1e-5))(shop_id_inp)

item_id_emb = Embedding(22170,15,input_length=1, embeddings_regularizer=l2(1e-5))(item_id_inp)

item_category_id_emb = Embedding(84,9,input_length=1, embeddings_regularizer=l2(1e-5))(item_category_id_inp)

month_emb = Embedding(12,4,input_length=1, embeddings_regularizer=l2(1e-5))(month_inp)

days_emb = Embedding(3,1,input_length=1, embeddings_regularizer=l2(1e-5))(days_inp)
x = concatenate([shop_id_emb,item_id_emb,item_category_id_emb,month_emb,days_emb])

x = Flatten()(x)





x = BatchNormalization()(x)

x = Dense(50,activation='relu')(x)

x = Dropout(0.2)(x)



x = BatchNormalization()(x)

x = Dense(10,activation='relu')(x)

x = Dropout(0.2)(x)



x = Dense(1,activation='relu')(x)

nn_embed_model = Model([shop_id_inp,item_id_inp,item_category_id_inp,month_inp,days_inp],x)

nn_embed_model.compile(loss = root_mean_squared_error,optimizer='adam')

nn_embed_model.summary()
nn_embed_model.fit(x_train_cat,

             y_train,

             epochs=5,

             validation_data=[x_valid_cat,y_valid],

             workers=4,

             use_multiprocessing=True,

             batch_size = 16384,

             callbacks = [EarlyStopping(patience=1,monitor='val_loss')]             

             )

#save_model(nn_embed_model,'nn_embed_model')
nn_embed_model = read_model('nn_embed_model')

y_test = nn_embed_model.predict(x_test_cat).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": y_test.flatten()

})

submission.to_csv('nn_embed_model_submission.csv', index=False)
data = pd.read_pickle('data3.pkl')

cat_feat = ['shop_id','item_id','item_category_id','month','days'] # Categorical features

noncat_feat = list(set(data.columns)-set(cat_feat)-set(['cnt'])) # Non-categorical features



x_train = data[data.date_block_num < 33][data.date_block_num != 10].drop('cnt', axis='columns')

y_train = data[data.date_block_num < 33][data.date_block_num != 10]['cnt']



x_valid = data[(data.date_block_num == 33) | (data.date_block_num == 10)].drop('cnt', axis='columns')

y_valid = data[(data.date_block_num == 33) | (data.date_block_num == 10)]['cnt']



x_test = data[data.date_block_num == 34].drop('cnt', axis='columns')

x_test = pd.merge(test,x_test,on=['item_id','shop_id'], right_index=True).sort_index() # This alignes x_test rows with test rows (for submission) 

x_test = x_test[x_train.columns]



x_train_cat = [x_train[feat] for feat in cat_feat]

x_valid_cat = [x_valid[feat] for feat in cat_feat]

x_test_cat =  [x_test[feat] for feat in cat_feat]



x_train_noncat = [np.reshape(np.array(x_train[feat]),(len(x_train[feat]),1,1)) for feat in noncat_feat]

x_valid_noncat = [np.reshape(np.array(x_valid[feat]),(len(x_valid[feat]),1,1)) for feat in noncat_feat]

x_test_noncat = [np.reshape(np.array(x_test[feat]),(len(x_test[feat]),1,1)) for feat in noncat_feat]
noncat_inps = []

for i in range(len(noncat_feat)):

  noncat_inps.extend([Input(shape=(1,1,),dtype='float32')]) 
shop_id_inp = Input(shape=(1,),dtype='int8')

item_id_inp = Input(shape=(1,),dtype='int16')

item_category_id_inp = Input(shape=(1,),dtype='int8')

month_inp = Input(shape=(1,),dtype='int8')

days_inp = Input(shape=(1,),dtype='int8')



shop_id_emb = Embedding(60,7,input_length=1, embeddings_regularizer=l2(1e-4))(shop_id_inp)

item_id_emb = Embedding(22170,15,input_length=1, embeddings_regularizer=l2(1e-4))(item_id_inp)

item_category_id_emb = Embedding(84,9,input_length=1, embeddings_regularizer=l2(1e-4))(item_category_id_inp)

month_emb = Embedding(12,4,input_length=1, embeddings_regularizer=l2(1e-4))(month_inp)

days_emb = Embedding(3,1,input_length=1, embeddings_regularizer=l2(1e-4))(days_inp)
x = concatenate([shop_id_emb,item_id_emb,item_category_id_emb,month_emb,days_emb]+noncat_inps)

x = Flatten()(x)



x = BatchNormalization()(x)

x = Dense(100,activation='relu')(x)

x = Dense(100,activation='relu')(x)

x = Dropout(0.3)(x)



x = BatchNormalization()(x)

x = Dense(50,activation='relu')(x)

x = Dense(50,activation='relu')(x)

x = Dropout(0.2)(x)



x = BatchNormalization()(x)

x = Dense(10,activation='relu')(x)

x = Dense(10,activation='relu')(x)

x = Dropout(0.2)(x)



x = Dense(1,activation='relu')(x)

nn_model = Model([shop_id_inp,item_id_inp,item_category_id_inp,month_inp,days_inp]+noncat_inps,x)

nn_model.compile(loss = 'mean_squared_error' ,optimizer='adam')

nn_model.summary()
np.random.seed(2020)

nn_model.fit(x_train_cat+x_train_noncat,

             y_train,

             epochs=4,

             validation_data=[x_valid_cat+x_valid_noncat ,y_valid],

             workers=4,

             use_multiprocessing=True,

             batch_size = 16384,

             shuffle = True,

             callbacks = [EarlyStopping(patience=1,monitor='val_loss')],

             initial_epoch = 3 

             )

#save_model(nn_model,'nn_model')
nn_model = read_model('nn_model')

y_test = nn_model.predict(x_test_cat+x_test_noncat).clip(0, 20)

submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": y_test.flatten()

})

submission.to_csv('nn_submission.csv', index=False)
data = pd.read_pickle('data3.pkl')

cat_feat = ['shop_id','item_id','item_category_id','month','days']

noncat_feat = list(set(data.columns)-set(cat_feat)-set(['cnt'])) # Non-categorical features

nn_model = read_model('nn_model')

embedded = pd.DataFrame({'n':[60,22170,84,12,3],'m':[7,15,9,4,1]}, index = ['shop_id','item_id','item_category_id','month','days'])
cat_to_vec = []

for i in range(len(cat_feat)):

  cat_to_vec.extend([nn_model.layers[i+5].get_weights()[0]])  
start  = 0

stop = 0

i=0

for feat in cat_feat:

  vectorize = cat_to_vec[i]

  i=i+1

  stop = stop + embedded.loc[feat,'m']

  a = pd.DataFrame(vectorize[:][data[feat]],columns=list(range(start,stop)))

  data = data.join(a,how = 'right')

  start = stop
x_train = data[data.date_block_num < 33][data.date_block_num>11].drop('cnt', axis='columns')

y_train = data[data.date_block_num < 33][data.date_block_num>11]['cnt']



x_valid = data[(data.date_block_num == 33) | (data.date_block_num == 10)].drop('cnt', axis='columns')

y_valid = data[(data.date_block_num == 33) | (data.date_block_num == 10)]['cnt']



x_test = data[data.date_block_num == 34]

x_test = pd.merge(test,x_test,on=['item_id','shop_id'], right_index=True).sort_index() # This alignes x_test rows with test rows (for submission) 

x_test = x_test[x_train.columns]
xgb_embed_model = XGBRegressor(

    max_depth=8,

    n_estimators=50,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)
xgb_embed_model = read_model_xgb('xgb_embed_model')

y_test = xgb_embed_model.predict(x_test).clip(0, 20)

submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": y_test.flatten()

})

submission.to_csv('xgb_embed_submission.csv', index=False)