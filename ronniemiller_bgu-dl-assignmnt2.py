# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

from keras.models import Model
from keras.layers import *
from keras.callbacks import *
from keras.regularizers import l2
from keras.optimizers import *
from keras.utils import to_categorical
import datetime
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
from keras import backend as K
from sklearn.model_selection import KFold


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# define function that calaculate RMSE
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
train_set = pd.read_csv("../input/bgu-dl-assignmnt2-features-extraction/train_set.csv")
test_set = pd.read_csv("../input/bgu-dl-assignmnt2-features-extraction/test_set.csv")
target = pd.read_csv("../input/bgu-dl-assignmnt2-features-extraction/target.csv", header=None)

print("shape of train : ",train_set.shape)
print("shape of test : ",test_set.shape)
print("shape of target : ",target.shape)
cat_col = ['feature_1','feature_2', 'feature_3', 'merchant_group_id', 'merchant_category_id', 'subsector_id', 'category_1',
          'most_recent_sales_range', 'most_recent_purchases_range', 'category_4', 'city_id', 'state_id', 'category_2']
numeric_col = train_set.columns[~train_set.columns.isin(np.append(cat_col, ['card_id', 'first_active_month']))]
used_col = np.concatenate((cat_col, numeric_col), axis=0)
def preprocess(trx_data):
    for cat_col_name in cat_col:
        lbl = LabelEncoder()
        lbl.fit(trx_data[cat_col_name].unique().astype('str'))
        trx_data[cat_col_name] = lbl.transform(trx_data[cat_col_name].astype('str'))
    
    for numeric_col_name in numeric_col:
        trx_data[numeric_col_name] = pd.to_numeric(trx_data[numeric_col_name])
        min_val = trx_data[numeric_col_name].min()
        max_val = trx_data[numeric_col_name].max()
        if min_val == max_val:
            trx_data[numeric_col_name] = 0
            print(numeric_col_name)
        else:
            trx_data[numeric_col_name] = (max_val - trx_data[numeric_col_name]) / (max_val - min_val)

    return trx_data

# remove nan values from data set
train_set_no_nan = train_set.fillna(-20)
test_set_no_nan = test_set.fillna(-20)

train_set = preprocess(train_set_no_nan)
test_set = preprocess(test_set_no_nan)
X_train, X_test, y_train, y_test = train_test_split(train_set, target, test_size=0.2, random_state=24)

print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
X_train.head()
X_test.head()
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train[used_col], y_train.iloc[:,0])

lgb_pred_test = lgb_model.predict(X_test[used_col])
lgb_pred_train = lgb_model.predict(X_train[used_col])

print('test RMSE:', mean_squared_error(y_test, lgb_pred_test) ** 0.5)
print('train RMSE:', mean_squared_error(y_train, lgb_pred_train) ** 0.5)
embedding_col = ['feature_1', 'feature_2', 'feature_3', 'year', 'month']
other_col = [x for x in used_col if x not in embedding_col]
f1_unique_val = len(train_set['feature_1'].unique())
f2_unique_val = len(train_set['feature_2'].unique())
f3_unique_val = len(train_set['feature_3'].unique())
year_unique_val = len(train_set['year'].unique())
month_unique_val = len(train_set['month'].unique())
f1_inp = Input(shape=(1,),dtype='int64')
f2_inp = Input(shape=(1,),dtype='int64')
f3_inp = Input(shape=(1,),dtype='int64')
year_inp = Input(shape=(1,),dtype='int64')
month_inp = Input(shape=(1,),dtype='int64')

f1_emb = Embedding(f1_unique_val,2,input_length=1, embeddings_regularizer=l2(1e-6))(f1_inp)
f2_emb = Embedding(f2_unique_val,1,input_length=1, embeddings_regularizer=l2(1e-6))(f2_inp)
f3_emb = Embedding(f3_unique_val,1,input_length=1, embeddings_regularizer=l2(1e-6))(f3_inp)
year_emb = Embedding(year_unique_val,3,input_length=1, embeddings_regularizer=l2(1e-6))(year_inp)
month_emb = Embedding(month_unique_val,4,input_length=1, embeddings_regularizer=l2(1e-6))(month_inp)
x = concatenate([f1_emb,f2_emb,f3_emb,year_emb,month_emb])
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(10,activation='relu')(x)
x = Dense(10,activation='relu')(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(10,activation='relu')(x)
x = Dense(10,activation='relu')(x)
x = Dropout(0.7)(x)
x = Dense(1, activation='sigmoid')(x) #activation='linear'
emb_model = Model([f1_inp,f2_inp,f3_inp,year_inp,month_inp],x)
#emb_model.compile(loss='mse',optimizer='adam')

emb_model.compile(optimizer="RMSProp", loss=root_mean_squared_error)

print(emb_model.summary())

emb_model.fit([X_train[col] for col in embedding_col], y_train, epochs=5)

emb_pred_test = emb_model.predict([X_test[col] for col in embedding_col])
emb_pred_train = emb_model.predict([X_train[col] for col in embedding_col])

print('test RMSE', mean_squared_error(y_test, emb_pred_test) ** 0.5)
print('train RMSE', mean_squared_error(y_train, emb_pred_train) ** 0.5)
emb_output = emb_model.layers[11].output

feature_model = Model(emb_model.input, emb_output)

feature_model.compile(optimizer = "RMSProp", loss = root_mean_squared_error)
print(feature_model.summary())

featurs = feature_model.predict([X_train[col] for col in embedding_col])
features_test = feature_model.predict([X_test[col] for col in embedding_col])

lgb_model = lgb.LGBMRegressor()
lgb_model.fit(featurs, y_train.values)

lgb_pred_test = lgb_model.predict(features_test)
lgb_pred_train = lgb_model.predict(featurs)

print('test RMSE', mean_squared_error(y_test, lgb_pred_test) ** 0.5)
print('train RMSE', mean_squared_error(y_train, lgb_pred_train) ** 0.5)
# define continuous input
continuous_input = Input(shape=(len(other_col),))

# define categorical input                         
f1_emb = Reshape((2,))(f1_emb)
f2_emb = Reshape((1,))(f2_emb)
f3_emb = Reshape((1,))(f3_emb)
year_emb = Reshape((3,))(year_emb)
month_emb = Reshape((4,))(month_emb)
                         
#split train set to train and validation set
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=6)
                         
# define function to create input to model
def get_input(data):
    inp = [data[other_col], data['feature_1'], data['feature_2'], data['feature_3'], data['year'], data['month']]
    return inp
x = concatenate([continuous_input,f1_emb,f2_emb,f3_emb,year_emb,month_emb])
x = BatchNormalization()(x)
x = Dense(10,activation='relu')(x)
x = Dense(10,activation='relu')(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(10,activation='relu')(x)
x = Dense(10,activation='relu')(x)
x = Dropout(0.7)(x)
x = Dense(1, activation='linear')(x)
emb_cont_model = Model([continuous_input,f1_inp,f2_inp,f3_inp,year_inp,month_inp],x)

rmsprop_opt = RMSprop(lr=0.005)
emb_cont_model.compile(optimizer = rmsprop_opt, loss = root_mean_squared_error)

print(emb_cont_model.summary())

'''def set_callbacks(description='run1',patience=15,tb_base_logdir='./logs/'):
    cp = ModelCheckpoint('best_model_weights_{}.h5'.format(description),save_best_only=True)
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=5)   
    tb = TensorBoard(log_dir='{}{}'.format(tb_base_logdir,description))
    cb = [cp,es,tb,rlop]
    return cb'''

history = emb_cont_model.fit(get_input(X_train_val), y_train_val, epochs=5, batch_size=16, 
          validation_data=(get_input(X_val), y_val)) #callbacks=set_callbacks()

emb_cont_pred_test = emb_cont_model.predict(get_input(X_test))
emb_cont_pred_train = emb_cont_model.predict(get_input(X_train_val))

print('test RMSE', mean_squared_error(y_test, emb_cont_pred_test) ** 0.5)
print('train RMSE', mean_squared_error(y_train_val, emb_cont_pred_train) ** 0.5)
