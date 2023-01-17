# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from keras.metrics import *
from keras.layers import *
from keras.models import Model,Sequential
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss
from keras.preprocessing.sequence import pad_sequences
import math
from keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.optimizers import *
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import *
path='/kaggle/input/competitive-data-science-predict-future-sales/'
categories = pd.read_csv(path + 'item_categories.csv')
items = pd.read_csv(path + 'items.csv')
sales_train = pd.read_csv(path + 'sales_train.csv')
test = pd.read_csv(path + 'test.csv')
shops = pd.read_csv(path + 'shops.csv')
sales_train
train = sales_train[sales_train['date_block_num']!=33]
val = sales_train[sales_train['date_block_num']==33]
y_train = sales_train['item_cnt_day']
X_train = sales_train.drop(['item_cnt_day','date'],axis = 1)
lr = LinearRegression()
score=cross_val_score(lr, X_train, y_train, cv=5,scoring='neg_mean_squared_error')
print("average RMSE:",str((-1*np.average(score))**0.5))
train_pre = sales_train[sales_train['item_cnt_day']>=0]
train_pre = train_pre[train_pre['item_price']>=0]
train_pre
y = train_pre['item_cnt_day']
X = train_pre.drop(['item_cnt_day','date'],axis = 1)
X_all_items = X['item_id'].append(test['item_id'])
X_all_shops = X['shop_id'].append(test['shop_id'])
items_en = {p:i for (i,p) in enumerate(X_all_items.unique())}
shops_en = {p:i for (i,p) in enumerate(X_all_shops.unique())}
date_block_nums_en = {p:i for (i,p) in enumerate(X['date_block_num'].unique())}
X_processed = X.loc[:,['item_id','shop_id','date_block_num']].copy()
X_processed['item_id'] = [items_en[x] for x in X['item_id']]
X_processed['shop_id'] = [shops_en[x] for x in X['shop_id']]
X_processed['date_block_num'] = [date_block_nums_en[x] for x in X['date_block_num']]
X_test_processed = test.loc[:,['item_id','shop_id']].copy()
X_test_processed['item_id'] = [items_en[x] for x in test['item_id']]
X_test_processed['shop_id'] = [shops_en[x] for x in test['shop_id']]
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.3)
items_inp = Input(shape=(1,),dtype='int64')
shops_inp = Input(shape=(1,),dtype='int64')
date_block_nums_inp = Input(shape=(1,),dtype='int64')

items_emb = Embedding(len(items_en),5,input_length=1, embeddings_regularizer=l2(1e-6))(items_inp)
shops_emb = Embedding(len(shops_en),5,input_length=1, embeddings_regularizer=l2(1e-6))(shops_inp)
date_block_nums_emb = Embedding(len(date_block_nums_en)+1,5,input_length=1, embeddings_regularizer=l2(1e-6))(date_block_nums_inp)
x = concatenate([items_emb,shops_emb,date_block_nums_emb])
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(100,activation='relu')(x)
x = Dense(10,activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(1,activation='relu')(x)
nn_model = Model([items_inp,shops_inp,date_block_nums_inp],x)
optimizer = RMSprop(lr=0.005)
nn_model.compile(loss = 'mse',optimizer=optimizer,metrics=[RootMeanSquaredError()])
nn_model.summary()
def set_callbacks(patience=15):
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=5)   
    cb = [es,rlop]
    return cb
history=nn_model.fit([X_train['item_id'],X_train['shop_id'],X_train['date_block_num']],y_train,epochs=15,
                 validation_data=[[X_val['item_id'],X_val['shop_id'],X_val['date_block_num']],y_val],batch_size = 4096, callbacks=set_callbacks())
def plot_history(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
plot_history(history)
X_test_processed['date_block_num'] = 34

predictions=nn_model.predict([X_test_processed['item_id'],X_test_processed['shop_id'],X_test_processed['date_block_num']])
preds = pd.DataFrame(predictions, columns=['item_cnt_month'])
preds.to_csv('submission.csv',index_label='ID')
train = sales_train.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_','item_name', 'shop_name', 'item_category_name'], axis=1)
test_joined = test.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_','item_name', 'shop_name', 'item_category_name'], axis=1)
test_joined
train_pre_f = train[sales_train['item_cnt_day']>=0]
train_pre_f = train_pre_f[train_pre_f['item_price']>=0]
train_pre_f
y_f = train_pre_f['item_cnt_day']
X_f = train_pre_f.drop(['item_cnt_day','date'],axis = 1)
X_all_categories = X_f['item_category_id'].append(test_joined['item_category_id'])
categories_en = {p:i for (i,p) in enumerate(X_all_categories.unique())}
X_processed_f = X_f.loc[:,['item_id','shop_id','date_block_num','item_category_id']].copy()
X_processed_f['item_id'] = [items_en[x] for x in X_f['item_id']]
X_processed_f['shop_id'] = [shops_en[x] for x in X_f['shop_id']]
X_processed_f['date_block_num'] = [date_block_nums_en[x] for x in X_f['date_block_num']]
X_processed_f['item_category_id'] = [categories_en[x] for x in X_f['item_category_id']]
X_test_processed_f = test_joined.loc[:,['item_id','shop_id','item_category_id']].copy()
X_test_processed_f['item_id'] = [items_en[x] for x in test_joined['item_id']]
X_test_processed_f['shop_id'] = [shops_en[x] for x in test_joined['shop_id']]
X_test_processed_f['item_category_id'] = [categories_en[x] for x in test_joined['item_category_id']]
X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(X_processed_f, y, test_size=0.3)
categories_inp = Input(shape=(1,),dtype='int64')

categories_emb = Embedding(len(categories_en),5,input_length=1, embeddings_regularizer=l2(1e-6))(categories_inp)
x = concatenate([items_emb,shops_emb,date_block_nums_emb,categories_emb])
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(100,activation='relu')(x)
x = Dense(10,activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(1,activation='relu')(x)
nn_model = Model([items_inp,shops_inp,date_block_nums_inp,categories_inp],x)
optimizer = RMSprop(lr=0.005)
nn_model.compile(loss = 'mse',optimizer=optimizer,metrics=[RootMeanSquaredError()])
nn_model.summary()
history=nn_model.fit([X_train_f['item_id'],X_train_f['shop_id'],X_train_f['date_block_num'],X_train_f['item_category_id']],y_train_f,epochs=15,
                 validation_data=[[X_val_f['item_id'],X_val_f['shop_id'],X_val_f['date_block_num'],X_val_f['item_category_id']],y_val_f],batch_size = 4096, callbacks=set_callbacks())
plot_history(history)
X_test_processed_f['date_block_num'] = 34

predictions=nn_model.predict([X_test_processed_f['item_id'],X_test_processed_f['shop_id'],X_test_processed_f['date_block_num'],X_test_processed_f['item_category_id']])
preds = pd.DataFrame(predictions, columns=['item_cnt_month'])
preds.to_csv('submissionF.csv',index_label='ID')
nn_model.layers.pop()
lr_2 = LinearRegression()

score = cross_val_score(lr_2, nn_model.predict([X_processed_f['item_id'],X_processed_f['shop_id'],X_processed_f['date_block_num'],X_processed_f['item_category_id']]), y, cv=5,scoring='neg_mean_squared_error')
print("average RMSE:",str((-1*np.average(score))**0.5))
lr_3 = LinearRegression()
lr_3.fit(nn_model.predict([X_processed_f['item_id'],X_processed_f['shop_id'],X_processed_f['date_block_num'],X_processed_f['item_category_id']]),y )
preds = lr_3.predict(nn_model.predict([X_test_processed_f['item_id'],X_test_processed_f['shop_id'],X_test_processed_f['date_block_num'],X_test_processed_f['item_category_id']]))
preds_3 = pd.DataFrame(preds, columns=['item_cnt_month'])
preds_3.to_csv('submissionLR.csv',index_label='ID')