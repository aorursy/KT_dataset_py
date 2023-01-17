import pandas as pd

import numpy as np

import os

print(os.listdir('../input/zaloprivate'))
train_df = pd.read_csv('../input/music-hit-prediction/train_info.tsv',delimiter='\t',encoding='utf-8')

train_label = pd.read_csv('../input/music-hit-prediction/train_rank.csv')

test_df = pd.read_csv('../input/zaloprivate/private_info.tsv',delimiter='\t',encoding='utf-8')

# train_audio = pd.read_csv('../input/zalo-ai-2019-hit-song-extract-features/train_extract_features.csv')

# test_audio = pd.read_csv('../input/zalo-ai-2019-hit-song-extract-features/test_extract_features.csv')

# y_audio = pd.read_csv('../input/music-hit-prediction/train_rank.csv')

# print(train_df.shape)

# print(test_df.shape)

# print(train_audio.shape)

# print(test_audio.shape)

# train_df.head()
# train_audio['label'] = train_audio['ID']

# train_audio['title'] = train_audio['ID'].astype('object')

# train_audio['artist_name'] = train_audio['ID'].astype('object')

# train_audio['artist_id'] = train_audio['ID'].astype('object')

# train_audio['composers_name'] = train_audio['ID'].astype('object')

# train_audio['composers_id'] = train_audio['ID'].astype('object')

# train_audio['release_time'] = train_audio['ID'].astype('object')
# for i in range(train_audio.shape[0]):

#     idx = train_audio.iloc[i]['ID'];

#     label = y_audio.loc[y_audio['ID'] == idx]['label']

#     title = train_df.loc[train_df['ID'] == idx]['title'][int(str(train_df.loc[train_df['ID'] == idx]['title']).split(' ')[0])]

#     artist_name = train_df.loc[train_df['ID'] == idx]['artist_name'][int(str(train_df.loc[train_df['ID'] == idx]['artist_name']).split(' ')[0])]

#     artist_id = train_df.loc[train_df['ID'] == idx]['artist_id'][int(str(train_df.loc[train_df['ID'] == idx]['artist_id']).split(' ')[0])]

#     composers_name = train_df.loc[train_df['ID'] == idx]['composers_name'][int(str(train_df.loc[train_df['ID'] == idx]['composers_name']).split(' ')[0])]

#     composers_id = train_df.loc[train_df['ID'] == idx]['composers_id'][int(str(train_df.loc[train_df['ID'] == idx]['composers_id']).split(' ')[0])]

#     release_time = train_df.loc[train_df['ID'] == idx]['release_time'][int(str(train_df.loc[train_df['ID'] == idx]['release_time']).split(' ')[0])]

#     train_audio.at[i,'label'] = label

#     train_audio.at[i,'title'] = str(title)

#     train_audio.at[i,'artist_name'] = artist_name

#     train_audio.at[i,'artist_id'] = artist_id

#     train_audio.at[i,'composers_name'] = composers_name

#     train_audio.at[i,'composers_id'] = composers_id

#     train_audio.at[i,'release_time'] = release_time
# test_audio = test_audio.sort_values(by=['ID'])
# test_audio.head()
label = train_label['label']
ind = []

for i in train_df['artist_id']:

    count1 = 0

    count2 = 0

    count1 = len(i.split('.'))

#     print(count1)

    if count1 == 1:

        count2 = len(i.split(','))

    ind.append(max(count1, count2))

#     ind.append(count)

        

indx = pd.DataFrame({'num_singer': ind})

# indx.shape



train_df = pd.concat([train_df, indx], axis = 1)

for i in range(1):

    train_df['singer_'+ str(i)] = train_df['num_singer']

for i in range(1):

    train_df['singer_'+ str(i)].values[:] = 0

    

for i in range(train_df.shape[0]):

    num = train_df.at[i,'num_singer']

    st = str(train_df.iloc[i]['artist_id'])

    if num == 1:

        train_df.at[i,'singer_0'] = st

    elif num == 2:

        train_df.at[i,'singer_0'] = st.split('.')[0]

    else:

        string = st.split(',')

        train_df.at[i,'singer_0'] = string[0]



ind = []

for i in test_df['artist_id']:

    count1 = 0

    count2 = 0

    count1 = len(i.split('.'))

#     print(count1)

    if count1 == 1:

        count2 = len(i.split(','))

    ind.append(max(count1, count2))

#     ind.append(count)

        

indx = pd.DataFrame({'num_singer': ind})

# indx.shape



test_df = pd.concat([test_df, indx], axis = 1)

for i in range(1):

    test_df['singer_'+ str(i)] = test_df['num_singer']

for i in range(1):

    test_df['singer_'+ str(i)].values[:] = 0

    

for i in range(test_df.shape[0]):

    num = test_df.at[i,'num_singer']

    st = str(test_df.iloc[i]['artist_id'])

    if num == 1:

        test_df.at[i,'singer_0'] = st

    elif num == 2:

        test_df.at[i,'singer_0'] = st.split('.')[0]

    else:

        string = st.split(',')

        test_df.at[i,'singer_0'] = string[0]



ind = []

for i in train_df['composers_id']:

    count1 = 0

    count2 = 0

    count1 = len(i.split('.'))

#     print(count1)

    if count1 == 1:

        count2 = len(i.split(','))

    ind.append(max(count1, count2))

#     ind.append(count)

        

indx = pd.DataFrame({'num_composers': ind})

# indx.shape

# indx['num_composers'].max()

train_df = pd.concat([train_df, indx], axis = 1)

for i in range(1):

    train_df['composers_'+ str(i)] = train_df['num_composers']

for i in range(1):

    train_df['composers_'+ str(i)].values[:] = 0

    

for i in range(train_df.shape[0]):

    num = train_df.at[i,'num_composers']

    st = str(train_df.iloc[i]['composers_id'])

    if num == 1:

        train_df.at[i,'composers_0'] = st

    elif num == 2:

        train_df.at[i,'composers_0'] = st.split('.')[0]

#         train_df.at[i,'composers_1'] = st.split('.')[1]

    else:

        string = st.split(',')

        train_df.at[i,'composers_0'] = string[0]



ind = []

for i in test_df['composers_id']:

    count1 = 0

    count2 = 0

    count1 = len(i.split('.'))

#     print(count1)

    if count1 == 1:

        count2 = len(i.split(','))

    ind.append(max(count1, count2))

#     ind.append(count)

        

indx = pd.DataFrame({'num_composers': ind})

# indx.shape

# indx['num_composers'].max()

test_df = pd.concat([test_df, indx], axis = 1)

for i in range(1):

    test_df['composers_'+ str(i)] = test_df['num_composers']

for i in range(1):

    test_df['composers_'+ str(i)].values[:] = 0



for i in range(test_df.shape[0]):

    num = test_df.at[i,'num_composers']

    st = str(test_df.iloc[i]['composers_id'])

    if num == 1:

        test_df.at[i,'composers_0'] = st

    elif num == 2:

        test_df.at[i,'composers_0'] = st.split('.')[0]

#         test_df.at[i,'composers_1'] = st.split('.')[1]

    else:

        string = st.split(',')

        test_df.at[i,'composers_0'] = string[0]
print(train_df.shape)

print(test_df.shape)
all_data = pd.concat((train_df.loc[:,'ID':'composers_0'],

                      test_df.loc[:,'ID':'composers_0']))
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocess text

print('Preprocessing text...')

cols = [

    'artist_name', 

    'composers_name', 

    'title'

]



for c_i, c in tqdm(enumerate(cols)):

    tfidf = TfidfVectorizer(

        sublinear_tf=True,

        analyzer='word',

        max_features=200 ,

        )

    tfidf.fit(all_data[c])

    tfidf_train = np.array(tfidf.transform(train_df[c]).toarray(), dtype=np.float16)

    tfidf_test = np.array(tfidf.transform(test_df[c]).toarray(), dtype=np.float16)



    for i in range(200):

        train_df[c + '_tfidf_' + str(i)] = tfidf_train[:, i]

        test_df[c + '_tfidf_' + str(i)] = tfidf_test[:, i]

        

    del tfidf, tfidf_train, tfidf_test

#     gc.collect(
all_data = pd.concat((train_df.loc[:,'ID':'title_tfidf_199'],

                      test_df.loc[:,'ID':'title_tfidf_199']))
all_data.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



le.fit(all_data['artist_id']) 

all_data['artist_id'] = le.transform(all_data['artist_id'])



le.fit(all_data['composers_id']) 

all_data['composers_id'] = le.transform(all_data['composers_id'])
all_data['release_time'] = pd.to_datetime(all_data['release_time'], errors='coerce')



all_data['hour'] = all_data['release_time'].dt.hour

all_data['dayofweek'] = all_data['release_time'].dt.dayofweek

all_data['quarter'] = all_data['release_time'].dt.quarter

all_data['month'] = all_data['release_time'].dt.month

all_data['year'] = all_data['release_time'].dt.year



all_data['dayofyear'] = all_data['release_time'].dt.dayofyear

all_data['dayofmonth'] = all_data['release_time'].dt.day

all_data['weekofyear'] = all_data['release_time'].dt.weekofyear



all_data.head()
all_data = all_data.drop(['title', 'composers_name', 'artist_name', 'release_time'], axis = 1) #'artist_id', 'composers_id'], axis = 1)

print(all_data.shape)

all_data.head()
train_features = all_data[:train_df.shape[0]]

train_targets = label

test_features = all_data[train_df.shape[0]:]
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
hyper_params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': ['rmse'],

    'learning_rate': 0.01,

    'feature_fraction': 0.85,

    'subsample': 0.8,

    'subsample_freq': 2,

    'verbose': 0,

    "max_depth": 40,

    "num_leaves": 250,  

    "max_bin": 512,

    "num_iterations": 10000,

}
train_targets = pd.DataFrame(train_targets)
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import math  

from sklearn.model_selection import KFold, StratifiedKFold



score = []

predict_val = pd.DataFrame(test_df['ID'])

skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=123)

skf.get_n_splits(train_features, train_targets)

oof_lgb_df = pd.DataFrame()

predictions = pd.DataFrame(test_df['ID'])

x_test = test_features.drop(['ID'], axis = 1)



for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):

    x_train, y_train = train_features.iloc[trn_idx], train_targets.iloc[trn_idx]['label']

    x_valid, y_valid = train_features.iloc[val_idx], train_targets.iloc[val_idx]['label']

    index = x_valid['ID']

    x_train = x_train.drop(['ID'], axis = 1)

    x_valid = x_valid.drop(['ID'], axis = 1)

    p_valid = 0

    yp = 0

    yv = 0

    gbm = lgb.LGBMRegressor(**hyper_params)

    gbm.fit(x_train, y_train,

        eval_set=[(x_valid, y_valid)],

        eval_metric='rmse',

        verbose = 500,

        early_stopping_rounds=100)

    score.append(math.sqrt(mean_squared_error(gbm.predict(x_valid), y_valid)))

    yp += gbm.predict(x_test)

    yv += gbm.predict(x_valid)

    fold_pred = pd.DataFrame({'ID': index,

                              'label':gbm.predict(x_valid)})

    oof_lgb_df = pd.concat([oof_lgb_df, fold_pred], axis=0)

    

    predictions['fold{}'.format(fold+1)] = yp

oof_lgb_df.head()
# oof_lgb_df.loc[oof_lgb_df['ID'] == 1073748245]
score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

oof_lgb_df = oof_lgb_df.sort_values('ID')

oof_lgb_df.head()
lgb_predict = pd.DataFrame()

lgb_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5']+predictions['fold6']+predictions['fold7']+predictions['fold8']+predictions['fold9']+predictions['fold10'])/10
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import math  

from xgboost import XGBRegressor

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

# import lightgbm as lgb



score = []

predict_val = pd.DataFrame(test_df['ID'])

skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=123)

skf.get_n_splits(train_features, train_targets)

oof_xgb_df = pd.DataFrame()

predictions = pd.DataFrame(test_df['ID'])

x_test = test_features.drop(['ID'], axis = 1)



for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):

    x_train, y_train = train_features.iloc[trn_idx], train_targets.iloc[trn_idx]['label']

    x_valid, y_valid = train_features.iloc[val_idx], train_targets.iloc[val_idx]['label']

    index = x_valid['ID']

    x_train = x_train.drop(['ID'], axis = 1)

    x_valid = x_valid.drop(['ID'], axis = 1)

    p_valid = 0

    yp = 0

    yv = 0

    xgb = XGBRegressor(max_depth = 8, n_estimators = 2000, n_jobs = 16, random_state = 4, subsample = 0.8, colsample_bytree = 0.7, max_bin = 16) # tree_method = 'gpu_hist', gpu_id = 0)

    xgb.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], eval_metric = ['rmse'], early_stopping_rounds = 70, verbose = 200)

    score.append(math.sqrt(mean_squared_error(xgb.predict(x_valid), y_valid)))

    yp += xgb.predict(x_test)

    yv += xgb.predict(x_valid)

    fold_pred = pd.DataFrame({'ID': index,

                              'label':xgb.predict(x_valid)})

    oof_xgb_df = pd.concat([oof_xgb_df, fold_pred], axis=0)

    predictions['fold{}'.format(fold+1)] = yp

score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

# xgb_predict = pd.DataFrame()

# xgb_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5'])/5

oof_xgb_df = oof_xgb_df.sort_values('ID')

oof_xgb_df.head()
xgb_predict = pd.DataFrame()

xgb_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5']+predictions['fold6']+predictions['fold7']+predictions['fold8']+predictions['fold9']+predictions['fold10'])/10
from catboost import Pool, CatBoostRegressor



from sklearn.metrics import mean_squared_error

import math  

from sklearn.model_selection import KFold, StratifiedKFold



score = []

predict_val = pd.DataFrame(test_df['ID'])

skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=123)

skf.get_n_splits(train_features, train_targets)

oof_cb_df = pd.DataFrame()

predictions = pd.DataFrame(test_df['ID'])

x_test = test_features.drop(['ID'], axis = 1)

cat_features = ['artist_id','singer_0','composers_0','composers_id','hour','dayofweek','quarter','month','year','dayofyear', 'dayofmonth', 'weekofyear']





for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):

    x_train, y_train = train_features.iloc[trn_idx], train_targets.iloc[trn_idx]['label']

    x_valid, y_valid = train_features.iloc[val_idx], train_targets.iloc[val_idx]['label']

    index = x_valid['ID']

    x_train = x_train.drop(['ID'], axis = 1)

    x_valid = x_valid.drop(['ID'], axis = 1)

    p_valid = 0

    yp = 0

    yv = 0

    cb = CatBoostRegressor(rsm=0.8, depth=10, learning_rate=0.5, eval_metric='RMSE', cat_features=cat_features)

    cb.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], early_stopping_rounds = 70, verbose = 50)

    score.append(math.sqrt(mean_squared_error(cb.predict(x_valid), y_valid)))

    yp += cb.predict(x_test)

    yv += cb.predict(x_valid)

    fold_pred = pd.DataFrame({'ID': index,

                              'label':cb.predict(x_valid)})

    oof_cb_df = pd.concat([oof_cb_df, fold_pred], axis=0)

    predictions['fold{}'.format(fold+1)] = yp

score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

oof_cb_df = oof_cb_df.sort_values('ID')

oof_cb_df.head()
cb_predict = pd.DataFrame()

cb_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5']+predictions['fold6']+predictions['fold7']+predictions['fold8']+predictions['fold9']+predictions['fold10'])/10
import tensorflow as tf

import pandas as pd

import os

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras import Sequential

from keras import layers

from keras import backend as K

from keras.layers.core import Dense

from keras import regularizers

from keras.layers import Dropout

from keras.constraints import max_norm
all_data.head()
# artist_id = pd.get_dummies(all_data['singer_0','singer_1','singer_2','singer_3','singer_4','singer_5','singer_6','singer_7','singer_8','singer_9','singer_10','singer_11',])

# composers_id = pd.get_dummies(all_data['composers_0','composers_1','composers_2','composers_3','composers_4','composers_5','composers_6'])

columns = ['singer_0','composers_0','hour','dayofweek','quarter','month','year','dayofyear', 'dayofmonth', 'weekofyear']#'singer_2','singer_3','singer_4','singer_5','singer_6','singer_7','singer_8','singer_9','singer_10','singer_11','composers_0','composers_1','composers_2','composers_3','composers_4','composers_5','composers_6']

for column in columns:

    dummies = pd.get_dummies(all_data[column])

    all_data[dummies.columns] = dummies
all_data.head()
all_data = all_data.drop(columns, axis=1)
from numpy import array

from keras.models import Sequential

from keras.layers import Dense

from matplotlib import pyplot

from keras import backend

 

def rmse(y_true, y_pred):

    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
all_data.head()
all_data = all_data.drop(['artist_id', 'composers_id'], axis = 1)

all_data.head()
ID = pd.DataFrame(all_data['ID'])
all_data = all_data.drop(['ID'], axis = 1)
from sklearn.decomposition import PCA



pca = pca = PCA(0.97)

principal_components = pca.fit_transform(all_data)

all_data = pd.DataFrame(data = principal_components)

scaler = StandardScaler()

scaler.fit(all_data)

all_data = pd.DataFrame(scaler.transform(all_data))
all_data.head()
train_features = all_data[:train_df.shape[0]]

train_targets = label

test_features = all_data[train_df.shape[0]:]

train_id = ID[:train_df.shape[0]]
train_features = pd.concat([train_id, train_features], axis = 1)

train_features.head()
input_dim = train_features.shape[1] - 1;


from keras.preprocessing.text import *

from keras.preprocessing import sequence



from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf

import pandas as pd

import os

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras import Sequential

from keras import layers

from keras import backend as K

from keras.layers.core import Dense

from keras import regularizers

from keras.layers import Dropout

from keras.constraints import max_norm

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold, RepeatedKFold



import os 

print(os.listdir('../input/'))
# Model

model = Sequential()

model.add(Dense(512, input_dim=input_dim, kernel_initializer='normal', activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(128, kernel_initializer='normal', activation='relu'))

model.add(Dropout(rate=0.3))

# model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# # model.add(Dropout(rate=0.2))

# model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# model.add(Dropout(rate=0.3))

# model.add(Dense(64, kernel_initializer='normal', activation='relu'))

# model.add(Dropout(rate=0.2))

# model.add(Dense(16, kernel_initializer='normal', activation='relu'))

# model.add(Dropout(rate=0.3))

model.add(Dense(16, kernel_initializer='normal', activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(1, kernel_initializer='normal', activation='relu'))

# Compile model

model.compile(loss=[rmse], optimizer='adam', metrics=[rmse])

model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau



checkpoint = ModelCheckpoint('feed_forward_model.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, 

                                   verbose=1, mode='min', epsilon=0.001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=10)

callbacks_list = [early, checkpoint, reduceLROnPlat]
train_targets = pd.DataFrame(train_targets)
from catboost import Pool, CatBoostRegressor



from sklearn.metrics import mean_squared_error

import math  

from sklearn.model_selection import KFold, StratifiedKFold



score = []

predict_val = pd.DataFrame(test_df['ID'])

skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=123)

skf.get_n_splits(train_features, train_targets)

oof_nn_df = pd.DataFrame()

predictions = pd.DataFrame(test_df['ID'])

x_test = test_features

cat_features = ['artist_id','singer_0','composers_0','composers_id','hour','dayofweek','quarter','month','year','dayofyear', 'dayofmonth', 'weekofyear']





for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):

    x_train, y_train = train_features.iloc[trn_idx], train_targets.iloc[trn_idx]['label']

    x_valid, y_valid = train_features.iloc[val_idx], train_targets.iloc[val_idx]['label']

    index = x_valid['ID']

    x_train = x_train.drop(['ID'], axis = 1)

    x_valid = x_valid.drop(['ID'], axis = 1)

    p_valid = 0

    yp = 0

    yv = 0

    model.compile(loss=[rmse], optimizer='adam', metrics=[rmse])

    model.fit(x_train, y_train, batch_size = 2048, epochs = 125,

          validation_data = (x_valid, y_valid),

          callbacks = callbacks_list)

    model.load_weights('feed_forward_model.h5')

    score.append(math.sqrt(mean_squared_error(model.predict(x_valid), y_valid)))

    yp += model.predict(x_test)

    yv += model.predict(x_valid)

    fold_pred = pd.DataFrame({'ID': index,

                              'label':np.array(model.predict(x_valid)).reshape((model.predict(x_valid).shape[0],))})

    oof_nn_df = pd.concat([oof_nn_df, fold_pred], axis=0)

    predictions['fold{}'.format(fold+1)] = yp

score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

nn_predict = pd.DataFrame()

nn_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5']+predictions['fold6']+predictions['fold7']+predictions['fold8']+predictions['fold9']+predictions['fold10'])/10

oof_nn_df = oof_nn_df.sort_values('ID')

oof_nn_df.head()
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error

# import math  

# from sklearn.model_selection import KFold, StratifiedKFold



# score = []

# predict_val = pd.DataFrame(test_df['ID'])

# skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=123)

# skf.get_n_splits(train_features, train_targets)

# oof_svr_df = pd.DataFrame()

# predictions = pd.DataFrame(test_df['ID'])

# x_test = test_features

# cat_features = ['artist_id','singer_0','composers_0','composers_id','hour','dayofweek','quarter','month','year','dayofyear', 'dayofmonth', 'weekofyear']





# for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):

#     x_train, y_train = train_features.iloc[trn_idx], train_targets.iloc[trn_idx]['label']

#     x_valid, y_valid = train_features.iloc[val_idx], train_targets.iloc[val_idx]['label']

#     index = x_valid['ID']

#     x_train = x_train.drop(['ID'], axis = 1)

#     x_valid = x_valid.drop(['ID'], axis = 1)

#     p_valid = 0

#     yp = 0

#     yv = 0

#     clf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

#     clf.fit(x_train, y_train)

#     score.append(math.sqrt(mean_squared_error(clf.predict(x_valid), y_valid)))

#     yp += model.predict(x_test)

#     yv += model.predict(x_valid)

#     fold_pred = pd.DataFrame({'ID': index,

#                               'label':clf.predict(x_valid)})

#     oof_svm_df = pd.concat([oof_nn_df, fold_pred], axis=0)

#     predictions['fold{}'.format(fold+1)] = yp

score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

svm_predict = pd.DataFrame()

svm_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5']+predictions['fold6']+predictions['fold7']+predictions['fold8']+predictions['fold9']+predictions['fold10'])/10

oof_svm_df = oof_nn_df.sort_values('ID')

oof_svm_df.head()
oof_data = None

oof_data = pd.DataFrame({ 'lgbm': oof_lgb_df['label'],

                          'xgb': oof_xgb_df['label'],

                          'cb':  oof_cb_df['label'],

#                           'svm': oof_svm_df['label'],

                          'nn': oof_nn_df['label'],

                          'label': train_targets['label']

})

oof_data.head()
oof_test = pd.DataFrame({ 'lgbm': lgb_predict['predict'],

                          'xgb': xgb_predict['predict'],

                          'cb':  cb_predict['predict'],

#                           'svm': svm_predict['predict'],

                          'nn': nn_predict['predict']

})

oof_test.head()
from catboost import Pool, CatBoostRegressor



from sklearn.metrics import mean_squared_error

import math  

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.linear_model import LinearRegression



train_features = oof_data.loc[:,'lgbm':'nn']

x_test = oof_test.loc[:,'lgbm':'nn']

score = []

predict_val = pd.DataFrame(test_df['ID'])

skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=123)

skf.get_n_splits(train_features, train_targets)

predictions = pd.DataFrame(test_df['ID'])

# cat_features = ['artist_id','singer_0','composers_0','composers_id','hour','dayofweek','quarter','month','year','dayofyear', 'dayofmonth', 'weekofyear']





for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):

    x_train, y_train = train_features.iloc[trn_idx], oof_data.iloc[trn_idx]['label']

    x_valid, y_valid = train_features.iloc[val_idx], oof_data.iloc[val_idx]['label']

    p_valid = 0

    yp = 0

    yv = 0

    regressor = LinearRegression()

    regressor.fit(x_train, y_train)

    score.append(math.sqrt(mean_squared_error(regressor.predict(x_valid), y_valid)))

    yp += regressor.predict(x_test)

    yv += regressor.predict(x_valid)

    oof_nn_df = pd.concat([oof_nn_df, fold_pred], axis=0)

    predictions['fold{}'.format(fold+1)] = yp
score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

predict = pd.DataFrame()

predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5']+predictions['fold6']+predictions['fold7']+predictions['fold8']+predictions['fold9']+predictions['fold10'])/10
mysubmit = pd.DataFrame({'ID': test_df['ID'],

                         'label': predict['predict']})

print(mysubmit.shape)

mysubmit.head()
mysubmit.to_csv('mysubmit.csv', index=False)