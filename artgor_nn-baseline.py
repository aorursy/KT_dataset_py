import numpy as np 
import pandas as pd 
import json
import bq_helper
from pandas.io.json import json_normalize
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error as mse_loss

from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields

def load_df(csv_path='../input/train.csv', JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']):

    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    return df
%%time
train = load_df("../input/train.csv")
%%time
test = load_df("../input/test.csv")
train['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
test['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
train['trafficSource.isTrueDirect'].fillna(False, inplace=True)
test['trafficSource.isTrueDirect'].fillna(False, inplace=True)

train['date'] = pd.to_datetime(train['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
test['date'] = pd.to_datetime(test['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
cols_to_drop = [col for col in train.columns if train[col].nunique(dropna=False) == 1]
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop([col for col in cols_to_drop if col in test.columns], axis=1, inplace=True)

#only one not null value
train.drop(['trafficSource.campaignCode'], axis=1, inplace=True)

for col in ['visitNumber', 'totals.hits', 'totals.pageviews', 'totals.transactionRevenue']:
    train[col] = train[col].astype(float)
    
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'])
train['totals.bounces'] = train['totals.bounces'].fillna(0)
train['totals.newVisits'] = train['totals.newVisits'].fillna(0)
train['totals.pageviews'] = train['totals.pageviews'].fillna(0)
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0)
train['trafficSource.adContent'] = train['trafficSource.adContent'].fillna(0)
train['trafficSource.keyword'] = train['trafficSource.keyword'].fillna(0)
train['trafficSource.adwordsClickInfo.adNetworkType'] = train['trafficSource.adwordsClickInfo.adNetworkType'].fillna(0)
train['trafficSource.adwordsClickInfo.gclId'] = train['trafficSource.adwordsClickInfo.gclId'].fillna(0)
train['trafficSource.adwordsClickInfo.page'] = train['trafficSource.adwordsClickInfo.page'].fillna(0)
train['trafficSource.adwordsClickInfo.slot'] = train['trafficSource.adwordsClickInfo.slot'].fillna(0)

test['totals.bounces'] = test['totals.bounces'].fillna(0)
test['totals.newVisits'] = test['totals.newVisits'].fillna(0)
test['totals.pageviews'] = test['totals.pageviews'].fillna(0)
test['trafficSource.adContent'] = test['trafficSource.adContent'].fillna(0)
test['trafficSource.keyword'] = test['trafficSource.keyword'].fillna(0)
test['trafficSource.adwordsClickInfo.adNetworkType'] = test['trafficSource.adwordsClickInfo.adNetworkType'].fillna(0)
test['trafficSource.adwordsClickInfo.gclId'] = test['trafficSource.adwordsClickInfo.gclId'].fillna(0)
test['trafficSource.adwordsClickInfo.page'] = test['trafficSource.adwordsClickInfo.page'].fillna(0)
test['trafficSource.adwordsClickInfo.slot'] = test['trafficSource.adwordsClickInfo.slot'].fillna(0)
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['weekday'] = train['date'].dt.weekday
train['weekofyear'] = train['date'].dt.weekofyear

test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['weekday'] = test['date'].dt.weekday
test['weekofyear'] = test['date'].dt.weekofyear

train['browser_category'] = train['device.browser'] + '_' + train['device.deviceCategory']
train['browser_operatingSystem'] = train['device.browser'] + '_' + train['device.operatingSystem']

test['browser_category'] = test['device.browser'] + '_' + test['device.deviceCategory']
test['browser_operatingSystem'] = test['device.browser'] + '_' + test['device.operatingSystem']

train['source_country'] = train['trafficSource.source'] + '_' + train['geoNetwork.country']
test['source_country'] = test['trafficSource.source'] + '_' + test['geoNetwork.country']
                                                                   
train['visitNumber'] = np.log1p(train['visitNumber'])
test['visitNumber'] = np.log1p(test['visitNumber'])

train['totals.hits'] = np.log1p(train['totals.hits'])
test['totals.hits'] = np.log1p(test['totals.hits'].astype(int))

train['totals.pageviews'] = np.log1p(train['totals.pageviews'].fillna(0))
test['totals.pageviews'] = np.log1p(test['totals.pageviews'].astype(float).fillna(0))
num_cols = ['visitNumber', 'totals.hits', 'totals.pageviews', 'month_unique_user_count', 'day_unique_user_count', 'mean_hits_per_day'
           'sum_pageviews_per_network_domain', 'sum_hits_per_network_domain', 'count_hits_per_network_domain', 'sum_hits_per_region',
           'sum_hits_per_day', 'count_pageviews_per_network_domain', 'mean_pageviews_per_network_domain', 'weekday_unique_user_count',
           'sum_pageviews_per_region', 'count_pageviews_per_region', 'mean_pageviews_per_region', 'user_pageviews_count', 'user_hits_count',
           'count_hits_per_region', 'mean_hits_per_region', 'user_pageviews_sum', 'user_hits_sum', 'user_pageviews_sum_to_mean',
            'user_hits_sum_to_mean', 'user_pageviews_to_region', 'user_hits_to_region', 'mean_pageviews_per_network_domain',
           'mean_hits_per_network_domain','totals.bounces', 'totals.newVisits']
num_cols = [col for col in num_cols if col in train.columns]
no_use = ["date", "fullVisitorId", "sessionId", "visitId", "visitStartTime", 'totals.transactionRevenue', 'trafficSource.referralPath']
cat_cols = [col for col in train.columns if col not in num_cols and col not in no_use]
max_values = {}
for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))
    max_values[col] = max(train[col].max(), test[col].max())  + 2
# printing because I'm too lazy to write everything by hand. Open output to see.
for col in cat_cols:
    n = col.replace('.', '_')
    print(f'{n} = Input(shape=[1], name="{col}")')
    print(f'emb_{n} = Embedding({max_values[col]}, {(np.min(max_values[col]+1)//2, 50)})({col})')
    print(',', n)
def model(dense_dim_1=128, dense_dim_2=64, dense_dim_3=32, dense_dim_4=16, 
dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.0001):

    #Inputs
    channelGrouping = Input(shape=[1], name="channelGrouping")
    device_browser = Input(shape=[1], name="device.browser")
    device_deviceCategory = Input(shape=[1], name="device.deviceCategory")
    device_operatingSystem = Input(shape=[1], name="device.operatingSystem")
    day = Input(shape=[1], name="day")
    geoNetwork_city = Input(shape=[1], name="geoNetwork.city")
    geoNetwork_continent = Input(shape=[1], name="geoNetwork.continent")
    geoNetwork_country = Input(shape=[1], name="geoNetwork.country")
    geoNetwork_metro = Input(shape=[1], name="geoNetwork.metro")
    geoNetwork_networkDomain = Input(shape=[1], name="geoNetwork.networkDomain")
    geoNetwork_region = Input(shape=[1], name="geoNetwork.region")
    geoNetwork_subContinent = Input(shape=[1], name="geoNetwork.subContinent")
    trafficSource_adContent = Input(shape=[1], name="trafficSource.adContent")
    trafficSource_adwordsClickInfo_adNetworkType = Input(shape=[1], name="trafficSource.adwordsClickInfo.adNetworkType")
    trafficSource_adwordsClickInfo_gclId = Input(shape=[1], name="trafficSource.adwordsClickInfo.gclId")
    trafficSource_adwordsClickInfo_isVideoAd = Input(shape=[1], name="trafficSource.adwordsClickInfo.isVideoAd")
    trafficSource_adwordsClickInfo_page = Input(shape=[1], name="trafficSource.adwordsClickInfo.page")
    trafficSource_adwordsClickInfo_slot = Input(shape=[1], name="trafficSource.adwordsClickInfo.slot")
    trafficSource_campaign = Input(shape=[1], name="trafficSource.campaign")
    trafficSource_keyword = Input(shape=[1], name="trafficSource.keyword")
    trafficSource_medium = Input(shape=[1], name="trafficSource.medium")
    trafficSource_source = Input(shape=[1], name="trafficSource.source")
    month = Input(shape=[1], name="month")
    weekday = Input(shape=[1], name="weekday")
    weekofyear = Input(shape=[1], name="weekofyear")
    browser_category = Input(shape=[1], name="browser_category")
    browser_operatingSystem = Input(shape=[1], name="browser_operatingSystem")
    source_country = Input(shape=[1], name="source_country")

    totals_pageviews = Input(shape=[1], name="totals.pageviews")
    totals_hits = Input(shape=[1], name="totals.hits")
    visitNumber = Input(shape=[1], name="visitNumber")
    
    
    #Embeddings layers

    emb_channelGrouping = Embedding(9, 5)(channelGrouping)
    emb_device_browser = Embedding(130, 50)(device_browser)
    emb_device_deviceCategory = Embedding(4, 3)(device_deviceCategory)
    emb_device_operatingSystem = Embedding(25, 13)(device_operatingSystem)
    emb_day = Embedding(32, 16)(day)
    emb_geoNetwork_city = Embedding(957, 50)(geoNetwork_city)
    emb_geoNetwork_continent = Embedding(7, 4)(geoNetwork_continent)
    emb_geoNetwork_country = Embedding(229, 50)(geoNetwork_country)
    emb_geoNetwork_metro = Embedding(124, 50)(geoNetwork_metro)
    emb_geoNetwork_networkDomain = Embedding(41983, 50)(geoNetwork_networkDomain)
    emb_geoNetwork_region = Embedding(484, 50)(geoNetwork_region)
    emb_geoNetwork_subContinent = Embedding(24, 12)(geoNetwork_subContinent)
    emb_trafficSource_adContent = Embedding(78, 39)(trafficSource_adContent)
    emb_trafficSource_adwordsClickInfo_adNetworkType = Embedding(5, 3)(trafficSource_adwordsClickInfo_adNetworkType)
    emb_trafficSource_adwordsClickInfo_gclId = Embedding(59010, 50)(trafficSource_adwordsClickInfo_gclId)
    emb_trafficSource_adwordsClickInfo_isVideoAd = Embedding(3, 3)(trafficSource_adwordsClickInfo_isVideoAd)
    emb_trafficSource_adwordsClickInfo_page = Embedding(13, 7)(trafficSource_adwordsClickInfo_page)
    emb_trafficSource_adwordsClickInfo_slot = Embedding(5, 3)(trafficSource_adwordsClickInfo_slot)
    emb_trafficSource_campaign = Embedding(36, 18)(trafficSource_campaign)
    emb_trafficSource_keyword = Embedding(5394, 50)(trafficSource_keyword)
    emb_trafficSource_medium = Embedding(8, 4)(trafficSource_medium)
    emb_trafficSource_source = Embedding(501, 50)(trafficSource_source)
    emb_month = Embedding(13, 7)(month)
    emb_weekday = Embedding(8, 4)(weekday)
    emb_weekofyear = Embedding(53, 27)(weekofyear)
    emb_browser_category = Embedding(175, 50)(browser_category)
    emb_browser_operatingSystem = Embedding(215, 50)(browser_operatingSystem)
    emb_source_country = Embedding(4480, 50)(source_country)

    concat_emb1 = concatenate([
           Flatten() (emb_channelGrouping),
            Flatten() (emb_device_deviceCategory),
            Flatten() (emb_device_operatingSystem),
            Flatten() (emb_day),
            Flatten() (emb_geoNetwork_continent),
            Flatten() (emb_geoNetwork_subContinent),
            Flatten() (emb_trafficSource_adContent),
            Flatten() (emb_trafficSource_adwordsClickInfo_adNetworkType),
            Flatten() (emb_trafficSource_adwordsClickInfo_isVideoAd),
            Flatten() (emb_trafficSource_adwordsClickInfo_page),
            Flatten() (emb_trafficSource_adwordsClickInfo_slot),
            Flatten() (emb_trafficSource_campaign),
            Flatten() (emb_trafficSource_medium),
            Flatten() (emb_month),
            Flatten() (emb_weekday),
            Flatten() (emb_weekofyear),
            Flatten() (emb_geoNetwork_region)
    ])
    
    categ = Dropout(dropout1)(Dense(dense_dim_1,activation='relu') (concat_emb1))
    categ = BatchNormalization()(categ)
    categ = Dropout(dropout2)(Dense(dense_dim_2,activation='relu') (categ))
    
    concat_emb2 = concatenate([
           Flatten() (emb_browser_category), 
            Flatten() (emb_browser_operatingSystem), 
            Flatten() (emb_source_country), 
            Flatten() (emb_device_browser), 
            Flatten() (emb_geoNetwork_city), 
            Flatten() (emb_trafficSource_source), 
            Flatten() (emb_trafficSource_keyword), 
            Flatten() (emb_trafficSource_adwordsClickInfo_gclId), 
            Flatten() (emb_geoNetwork_networkDomain), 
            Flatten() (emb_geoNetwork_country), 
            Flatten() (emb_geoNetwork_metro), 
            Flatten() (emb_geoNetwork_region)
    ])
    categ2 = Dropout(dropout1)(Dense(dense_dim_1* 2,activation='relu') (concat_emb2))
    categ2 = BatchNormalization()(categ2)
    categ2 = Dropout(dropout2)(Dense(dense_dim_2* 2,activation='relu') (categ2))
    
    #main layer
    main_l = concatenate([
          categ
        , categ2
        , totals_pageviews
        , totals_hits
        , visitNumber
    ])
    
    main_l = Dropout(dropout3)(Dense(dense_dim_3,activation='relu') (main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dropout4)(Dense(dense_dim_4,activation='relu') (main_l))
    
    #output
    output = Dense(1) (main_l)

    model = Model([channelGrouping,
                   device_browser,
                   device_deviceCategory,
                   device_operatingSystem,
                   geoNetwork_city,
                   geoNetwork_continent,
                   geoNetwork_country,
                   geoNetwork_metro,
                   geoNetwork_networkDomain,
                   geoNetwork_region,
                   geoNetwork_subContinent,
                   trafficSource_adContent,
                   trafficSource_adwordsClickInfo_adNetworkType,
                   trafficSource_adwordsClickInfo_gclId,
                   trafficSource_adwordsClickInfo_isVideoAd,
                   trafficSource_adwordsClickInfo_page,
                   trafficSource_adwordsClickInfo_slot,
                   trafficSource_campaign,
                   trafficSource_keyword,
                   trafficSource_medium,
                   trafficSource_source,
                   month,
                   day,
                   weekday,
                   weekofyear,
                   browser_category,
                   browser_operatingSystem,
                   source_country,
                   totals_pageviews, totals_hits, visitNumber], output)
    #model = Model([**params], output)
    model.compile(optimizer = Adam(lr=lr),
                  loss= mse_loss,
                  metrics=[root_mean_squared_error])
    return model

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))
#model=model()
train = train.sort_values('date')
X = train.drop(no_use, axis=1)
y = train['totals.transactionRevenue']
X_test = test.drop([col for col in no_use if col in test.columns], axis=1)
n_fold = 10
folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)
# converting data to format which can be used by Keras
def get_keras_data(df, num_cols, cat_cols):
    cols = num_cols + cat_cols

    X = {col: np.array(df[col]) for col in cols}
    # print("Data ready for Vectorization")
    
    return X
X_test_keras = get_keras_data(X_test, num_cols, cat_cols)
def train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, reduce_lr=False, patience=3):
    """
    Helper function to train model. Also I noticed that ReduceLROnPlateau is rarely
    useful, so added an option to turn it off.
    """
    
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    model_checkpoint = ModelCheckpoint("model.hdf5",
                                       save_best_only=True, verbose=1, monitor='val_root_mean_squared_error', mode='min')
    if reduce_lr:
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.000005, verbose=1)
        hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(X_v, y_valid), verbose=False,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr])
    
    else:
        hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(X_v, y_valid), verbose=False,
                            callbacks=[early_stopping, model_checkpoint])

    keras_model = load_model("model.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})
    
    return keras_model
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
batch_size = 2 ** 10
epochs = 100
scores = []
for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    print('Fold:', fold_n)
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    X_t = get_keras_data(X_train, num_cols, cat_cols)
    X_v = get_keras_data(X_valid, num_cols, cat_cols)
    
    keras_model = model(dense_dim_1=64, dense_dim_2=64, dense_dim_3=32, dense_dim_4=8, 
                        dropout1=0.05, dropout2=0.1, dropout3=0.1, dropout4=0.05, lr=0.0001)
    mod = train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, reduce_lr=False, patience=3)
    oof[valid_index] = mod.predict(X_v).reshape(-1,)
    
    y_pred = mod.predict(X_test_keras).reshape(-1,)
    predictions += y_pred
    
    y_pred_valid = mod.predict(X_v)
    scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)
    print('Validation score: {}.'.format(scores[-1]))
    print('*'* 50)

predictions /= n_fold
3.7187107900597733 ** 0.5
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
submission = test[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = predictions
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv(f'nn_cv{np.mean(scores):.4f}_std_{np.std(scores):.4f}_prediction.csv', index=False)
oof_df = pd.DataFrame({"fullVisitorId": train["fullVisitorId"], "PredictedLogRevenue": oof})
oof_df.to_csv(f'nn_cv{np.mean(scores):.4f}_std_{np.std(scores):.4f}_oof.csv', index=False)



