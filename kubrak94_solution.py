import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import f_regression, SelectKBest 

import reverse_geocoder as revgc
from datetime import date, timedelta

from fastai.structured import *
from fastai.column_data import *

np.random.seed(42)
sns.set()
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
data_dir = 'data/'
train_data = pd.read_csv(data_dir + 'train_data.csv',
                         parse_dates=['rec_date'])
test_data = pd.read_csv(data_dir + 'test_data.csv',
                        parse_dates=['rec_date'])
receipts_history = pd.read_csv(data_dir + 'receipts_history.csv',
                              parse_dates=['rec_date'])
categories = pd.read_csv(data_dir + 'categories.csv')
target_column_names = list(set(train_data.columns) - set(test_data.columns))
train_column_names = test_data.columns
X = pd.concat([train_data[train_column_names], receipts_history[train_column_names], test_data], axis='rows')
y = pd.concat([train_data[target_column_names], receipts_history[target_column_names]])
for t in [X, y]:
    display(DataFrameSummary(t).summary())
X.sort_values(['shop_geo_lat'])
X[X.shop_geo_lat < 40]
X[(X.shop_geo_lon < 27) & (X.shop_geo_lon > 23)]
X[(X.shop_geo_lon < 19)]
X[(X.shop_geo_lat < 50) & (X.shop_geo_lon > 50) & (X.shop_geo_lon < 130)]
guys_with_strange_coordinates = X[(X.shop_geo_lat < 40) | (X.shop_geo_lon < 19)].user_id.unique()
len(guys_with_strange_coordinates)
avg_lat = X[(X.shop_geo_lat > 40) & (X.shop_geo_lon > 19)].shop_geo_lat.mean()
avg_lon = X[(X.shop_geo_lat > 40) & (X.shop_geo_lon > 19)].shop_geo_lon.mean()

avg_lat, avg_lon
def fill_coords(df, u):
    try:
        pair_coords = X[(X.user_id == u) & (X.shop_geo_lat > 40) & (X.shop_geo_lon > 19)].\
        groupby(['shop_geo_lat', 'shop_geo_lon']).aggregate('count').\
        sort_values('user_id', ascending=[False]).iloc[0].name
    except IndexError:
        pair_coords = (avg_lat, avg_lon)
    
    df.loc[(df['user_id'] == u) & (df['shop_geo_lat'] < 40), 'shop_geo_lat']  = pair_coords[0]
    df.loc[(df['user_id'] == u) & (df['shop_geo_lon'] < 19), 'shop_geo_lon']  = pair_coords[1]
    
    return df
for u in guys_with_strange_coordinates:
    X = fill_coords(X, u)
    train_data = fill_coords(train_data, u)
    test_data = fill_coords(test_data, u)
    receipts_history = fill_coords(receipts_history, u)
for t in [train_data, test_data, receipts_history, X]:
    display(t[(t.shop_geo_lat < 40) | (t.shop_geo_lon < 19)])
receipts_history.to_csv('data/receipts_history_clean.csv')
train_data.to_csv('data/train_data_clean.csv')
test_data.to_csv('data/test_data_clean.csv')
for t in [X, y]:
    display(DataFrameSummary(t).summary())
train_data = pd.read_csv(data_dir + 'train_data_clean.csv',
                         parse_dates=['rec_date'])
train_data = train_data.drop(columns="Unnamed: 0")
test_data = pd.read_csv(data_dir + 'test_data_clean.csv',
                        parse_dates=['rec_date'])
receipts_history = pd.read_csv(data_dir + 'receipts_history_clean.csv',
                              parse_dates=['rec_date'])
X_train = train_data.drop(columns=target_column_names)
X_test = test_data

y_train = train_data[target_column_names]

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                      test_size=0.3,
                                                      random_state=42)
X_train = pd.concat([X_train, receipts_history[train_column_names]], axis='rows')
y_train = pd.concat([y_train, receipts_history[target_column_names]], axis = 'rows')
add_datepart(X_train, 'rec_date', drop=False)
add_datepart(X_valid, 'rec_date', drop=False)
add_datepart(X_test, 'rec_date', drop=False)
from datetime import date, timedelta

#Russian public holidays
public_holidays = [date(2017, 1, 2),
                   date(2017, 1, 3),
                   date(2017, 1, 4),
                   date(2017, 1, 5),
                   date(2017, 1, 6),
                   date(2017, 2, 23),
                   date(2017, 2, 24),
                   date(2017, 3, 8),
                   date(2017, 5, 1),
                   date(2017, 5, 8),
                   date(2017, 5, 9),
                   date(2017, 6, 12),
                   date(2017, 11, 6),
                   date(2018, 1, 1),
                   date(2018, 1, 2),
                   date(2018, 1, 3),
                   date(2018, 1, 4),
                   date(2018, 1, 5),
                   date(2018, 1, 8),
                   date(2018, 2, 23),
                   date(2018, 3, 8),
                   date(2018, 3, 9),
                   date(2018, 4, 30),
                   date(2018, 5, 1),
                   date(2018, 5, 2),
                   date(2018, 5, 9),
                   date(2018, 6, 11),
                   date(2018, 6, 12),
                   date(2018, 11, 5),
                   date(2018, 12, 31)]

#Russian workdays of the weekends
non_holidays = [date(2018, 4, 28),
                date(2018, 6, 9),
                date(2018, 12, 29)]

public_holidays = [np.datetime64(i) for i in public_holidays]

non_holidays = [np.datetime64(i) for i in non_holidays]
def is_hol(row):
    return 1 if row['the_date'] in public_holidays else 0

def is_weekend(row):
    return 1 if row['the_date'] not in non_holidays and row['rec_Dayofweek'] in [5, 6] else 0

def is_friday(row):
    return 1 if ((row['the_date'] + timedelta(days=1) not in non_holidays and row['rec_Dayofweek'] + 1 in [5, 6]) \
                  or row['the_date'] + timedelta(days=1) in public_holidays) \
                  and row['the_date'] not in public_holidays and (row['rec_Dayofweek'] not in [5, 6] \
                  or row['the_date'] in non_holidays) \
            else 0
def add_holidays(df):
    df['the_date'] = pd.to_datetime(df.rec_date).dt.date
    df['is_holiday'] = df.apply(lambda row: is_hol(row), axis=1)
    df['is_weekend'] = df.apply(lambda row: is_weekend(row), axis=1)
    df['is_pre_holiday'] = df.apply(lambda row: is_friday(row), axis=1)
    
    df['the_date'] = df.apply(lambda row:np.datetime64(row.the_date), axis=1)
    
    return df
X_train = add_holidays(X_train)
X_valid = add_holidays(X_valid)
X_test = add_holidays(X_test)
def add_a_city(df):
    df['position_info'] = revgc.search(list(zip(df.shop_geo_lat, df.shop_geo_lon)))
    df['city'] = df.apply(lambda row: row.position_info['name'] if row.position_info['admin1'] \
                       not in ['Moscow', 'St.-Petersburg'] else row.position_info['admin1'], axis=1)
    df['country'] = df.apply(lambda row: row.position_info['cc'], axis=1)

    df = df.drop(columns=['position_info'])
    
    return df
X_train = add_a_city(X_train)
X_valid = add_a_city(X_valid)
X_test = add_a_city(X_test)
for t in [X_train, X_valid, X_test]:
    display(t[(t.country != 'RU') & (t.country != 'UA')])
display(X_test[X_test.user_id == 4298])
display(X_train[X_train.user_id == 4298])
X_train.set_value(1191, 'shop_geo_lat', X_train.iloc[13113 + int(len(train_data) * 0.7)].shop_geo_lat)
X_train.set_value(1191, 'shop_geo_lon', X_train.iloc[13113 + int(len(train_data) * 0.7)].shop_geo_lon)
X_train.set_value(1191, 'city', X_train.iloc[13113 + int(len(train_data) * 0.7)].city)
display(X_train[X_train.user_id == 4298])
def get_elapsed(df, fld):
    day1 = np.timedelta64(1, 'D')
    i = 0
    cur_date = public_holidays[i]
    next_date = public_holidays[i + 1]
    res_before = []
    res_after = []

    for v,d in zip(df[fld].values, df.the_date.values):
        
        while next_date < d:
            i += 1
            cur_date = public_holidays[i]
            next_date = public_holidays[i + 1]
            
        res_before.append(((d-cur_date).astype('timedelta64[D]') / day1) * (1 - v))
        res_after.append(((next_date-d).astype('timedelta64[D]') / day1) * (1 - v))
        
    df["Before"+fld.replace('is', '')] = res_before
    df["After"+fld.replace('is', '')] = res_after

def drop_columns(df):
    return df.drop(columns=['the_date', 'rec_date', 'rec_Elapsed', 'country'])
fld = 'is_holiday'
X_train = X_train.sort_values(['the_date'])
get_elapsed(X_train, fld)
fld = 'is_holiday'
X_valid = X_valid.sort_values(['the_date'])
get_elapsed(X_valid, fld)
fld = 'is_holiday'
X_test = X_test.sort_values(['the_date'])
get_elapsed(X_test, fld)
X_train = drop_columns(X_train)
X_valid = drop_columns(X_valid)
X_test = drop_columns(X_test)
X_train.head()
cities = pd.concat([X_train.city, X_valid.city, X_test.city], axis='rows')
city_encoder = LabelEncoder()
city_encoder.fit(cities)
X_train.city = city_encoder.transform(X_train.city)
X_valid.city = city_encoder.transform(X_valid.city)
X_test.city = city_encoder.transform(X_test.city)
from catboost import CatBoostRegressor, Pool
cat_features = [i for i in range(21) if i not in [1, 2, 19, 20]]
def get_selected_cat_features(cat_features):
    cf = []
    i = 0
    for k,j in enumerate(selector.get_support()):
        if j and k in cat_features:
            cf.append(i)
            i += 1
        elif j:
            i += 1
    return cf
preds = pd.DataFrame()
model = CatBoostRegressor(loss_function='RMSE', random_seed=0, verbose=False)

for j, i in enumerate(y_train.columns):
    print(j, end=' ')
    selector = SelectKBest(f_regression, k=15).fit(X_train, y_train[i])
    X_train_new = selector.transform(X_train)
    cf = get_selected_cat_features(cat_features)
    #train_pool = Pool(X_train, label=y_train[i], cat_features=cat_features)
    train_pool = Pool(X_train_new, label=y_train[i], cat_features=cf)
    X_valid_new = selector.transform(X_valid)
    #valid_pool = Pool(X_valid, cat_features=cat_features)
    valid_pool = Pool(X_valid_new, cat_features=cf)
    model.fit(train_pool)
    preds[i] = model.predict(valid_pool)
# All features
mean_squared_error(preds, y_valid)
# Select 15 features
mean_squared_error(preds, y_valid)
import xgboost as xg
preds = pd.DataFrame()
model = xg.XGBRegressor(objective="reg:linear", colsample_bytree=0.1, 
                         learning_rate=0.125, max_depth=4, alpha=10, n_estimators=20, eta=0.02, 
                        nrounds = 2000
                       )

for i in y_train.columns:
    selector = SelectKBest(f_regression, k=15).fit(X_train, y_train[i])
    X_train_new = selector.transform(X_train)
    model.fit(X_train_new, y_train[i])
    X_valid_new = selector.transform(X_valid)
    preds[i] = model.predict(X_valid_new)
mean_squared_error(preds, y_valid)
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape, Dropout
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
users = pd.concat([X_train.user_id, X_valid.user_id, X_test.user_id], axis='rows')
user_encoder = LabelEncoder()
user_encoder.fit(users)
X_train.user_id = user_encoder.transform(X_train.user_id)
X_valid.user_id = user_encoder.transform(X_valid.user_id)
X_test.user_id = user_encoder.transform(X_test.user_id)
X = pd.concat([X_train, X_valid, X_test], axis='rows')
contin_vars = ['shop_geo_lon', 'shop_geo_lat', 'Before_holiday', 'After_holiday']
cat_vars = list(set(X_train.columns) - set(contin_vars))
def get_embedding_size(n):
    return min(50, (n + 1) // 2)
for col in list(set(cat_vars) - set(['user_id', 'city'])):
    encoder = LabelEncoder()
    encoder.fit(X[col])
    X_train[col] = encoder.transform(X_train[col])
    X_valid[col] = encoder.transform(X_valid[col])
    X_test[col] = encoder.transform(X_test[col])
input_model = []
output_embeddings = []

# Forming the Embeddings
for column in cat_vars:
    unique_number = len(X[column].unique())
    input_ = Input(shape=(1,))
    embed_name = column+'_embedding'
    output_ = Embedding(unique_number, get_embedding_size(unique_number), name=embed_name)(input_)
    output_ = Reshape(target_shape=(get_embedding_size(unique_number),))(output_)
    
    input_model.append(input_)
    output_embeddings.append(output_)

output_model = Concatenate()(output_embeddings)
output_model = Dropout(0.9)(output_model)
output_model = Dense(1000, kernel_initializer="uniform")(output_model)
output_model = Activation('relu')(output_model)
output_model = Dense(500, kernel_initializer="uniform")(output_model)
output_model = Activation('relu')(output_model)
output_model = Dropout(0.5)(output_model)
output_model = Dense(25)(output_model)
output_model = Activation('sigmoid')(output_model)

model = Model(inputs=input_model, outputs=output_model)

model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()
def split_features(X_in):
    X_list = []
    
    for column in cat_vars:
        X_list.append(X_in[column])
        
    return X_list
X_train_split = split_features(X_train)
X_valid_split = split_features(X_valid)
model.fit(X_train_split, y_train, 
          validation_data=(X_valid_split, y_valid), 
          epochs=20, batch_size=len(X_train))
result = model.predict(split_features(X_valid))
mean_squared_error(result, y_valid)
mean_squared_error(np.zeros(y_valid.shape), y_valid)
train_data = pd.read_csv(data_dir + 'train_data_clean.csv',
                         parse_dates=['rec_date'])
train_data = train_data.drop(columns="Unnamed: 0")
test_data = pd.read_csv(data_dir + 'test_data_clean.csv',
                        parse_dates=['rec_date'])
receipts_history = pd.read_csv(data_dir + 'receipts_history_clean.csv',
                              parse_dates=['rec_date'])
X_train = train_data.drop(columns=target_column_names)
X_test = test_data

y_train = train_data[target_column_names]

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                      test_size=0.3,
                                                      random_state=42)
X_train = pd.concat([X_train, receipts_history[train_column_names]], axis='rows')
y_train = pd.concat([y_train, receipts_history[target_column_names]], axis = 'rows')
def generate_features(df):
    df['rec_hours'] = df.rec_date.dt.hour
    df['rec_minutes'] = df.rec_date.dt.minute
    df['rec_minutes_from_midnight'] = (
        df.rec_hours * 60 + df.rec_minutes 
    )
    df['rec_month'] = df.rec_date.dt.month 
    df['rec_dayofweek'] = df.rec_date.dt.dayofweek
    
    df['timestamp'] = (df.rec_date - pd.Timestamp("1970-01-01")) \
                        // pd.Timedelta('1s')
    
    
    df = df.drop(columns=['rec_date'])
    return df
X_train = generate_features(X_train)
X_valid = generate_features(X_valid)
X_test = generate_features(X_test)
preds_test = pd.DataFrame()
model = CatBoostRegressor(loss_function='RMSE', random_seed=0, verbose=False)

X_train_full = pd.concat([X_train, X_valid], axis='rows')
y_train_full = pd.concat([y_train, y_valid], axis='rows')

for j, i in tqdm(enumerate(y_train.columns)):
    print(j, end=' ')
    cat_features = [0, 3, 4, 6, 7]
    train_pool = Pool(X_train, label=y_train[i], cat_features=cat_features)
    valid_pool = Pool(X_test, cat_features=cat_features)
    model.fit(train_pool)
    preds_test[i] = model.predict(valid_pool)
preds_test['user_id'] = X_test.user_id
preds_test.to_csv('sample_submission_cat_boost.csv', index=None)