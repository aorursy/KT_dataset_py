

import numpy as np

import pandas as pd

#import matplotlib.pyplot as plt

#import seaborn as sns

from sklearn import model_selection, preprocessing

#import xgboost as xgb

import datetime

import lightgbm as lgbm

#load files

train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])

id_test = test.id



#multiplier = 0.969



#clean data

bad_index = train[train.life_sq > train.full_sq].index

train.loc[bad_index, "life_sq"] = np.NaN

equal_index = [601,1896,2791]

test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]

bad_index = test[test.life_sq > test.full_sq].index

test.loc[bad_index, "life_sq"] = np.NaN

bad_index = train[train.life_sq < 5].index

train.loc[bad_index, "life_sq"] = np.NaN

bad_index = test[test.life_sq < 5].index

test.loc[bad_index, "life_sq"] = np.NaN

bad_index = train[train.full_sq < 5].index

train.loc[bad_index, "full_sq"] = np.NaN

bad_index = test[test.full_sq < 5].index

test.loc[bad_index, "full_sq"] = np.NaN

kitch_is_build_year = [13117]

train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]

bad_index = train[train.kitch_sq >= train.life_sq].index

train.loc[bad_index, "kitch_sq"] = np.NaN

bad_index = test[test.kitch_sq >= test.life_sq].index

test.loc[bad_index, "kitch_sq"] = np.NaN

bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index

train.loc[bad_index, "kitch_sq"] = np.NaN

bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index

test.loc[bad_index, "kitch_sq"] = np.NaN

bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index

train.loc[bad_index, "full_sq"] = np.NaN

bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index

test.loc[bad_index, "full_sq"] = np.NaN

bad_index = train[train.life_sq > 300].index

train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN

bad_index = test[test.life_sq > 200].index

test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN

train.product_type.value_counts(normalize= True)

test.product_type.value_counts(normalize= True)

bad_index = train[train.build_year < 1500].index

train.loc[bad_index, "build_year"] = np.NaN

bad_index = test[test.build_year < 1500].index

test.loc[bad_index, "build_year"] = np.NaN

bad_index = train[train.num_room == 0].index

train.loc[bad_index, "num_room"] = np.NaN

bad_index = test[test.num_room == 0].index

test.loc[bad_index, "num_room"] = np.NaN

bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]

train.loc[bad_index, "num_room"] = np.NaN

bad_index = [3174, 7313]

test.loc[bad_index, "num_room"] = np.NaN

bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index

train.loc[bad_index, ["max_floor", "floor"]] = np.NaN

bad_index = train[train.floor == 0].index

train.loc[bad_index, "floor"] = np.NaN

bad_index = train[train.max_floor == 0].index

train.loc[bad_index, "max_floor"] = np.NaN

bad_index = test[test.max_floor == 0].index

test.loc[bad_index, "max_floor"] = np.NaN

bad_index = train[train.floor > train.max_floor].index

train.loc[bad_index, "max_floor"] = np.NaN

bad_index = test[test.floor > test.max_floor].index

test.loc[bad_index, "max_floor"] = np.NaN

train.floor.describe(percentiles= [0.9999])

bad_index = [23584]

train.loc[bad_index, "floor"] = np.NaN

train.material.value_counts()

test.material.value_counts()

train.state.value_counts()

bad_index = train[train.state == 33].index

train.loc[bad_index, "state"] = np.NaN

test.state.value_counts()



# brings error down a lot by removing extreme price per sqm

train.loc[train.full_sq == 0, 'full_sq'] = 50

train = train[train.price_doc/train.full_sq <= 600000]

train = train[train.price_doc/train.full_sq >= 10000]



# Add month-year

month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

train['month_year_cnt'] = month_year.map(month_year_cnt_map)



month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

test['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

train['week_year_cnt'] = week_year.map(week_year_cnt_map)



week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

test['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

train['month'] = train.timestamp.dt.month

train['dow'] = train.timestamp.dt.dayofweek



test['month'] = test.timestamp.dt.month

test['dow'] = test.timestamp.dt.dayofweek



# Other feature engineering

train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)

train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)



test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)

test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)



train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)

test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)



train['room_size'] = train['life_sq'] / train['num_room'].astype(float)

test['room_size'] = test['life_sq'] / test['num_room'].astype(float)



rate_2016_q2 = 1

rate_2016_q1 = rate_2016_q2 / .99903

rate_2015_q4 = rate_2016_q1 / .9831

rate_2015_q3 = rate_2015_q4 / .9834

rate_2015_q2 = rate_2015_q3 / .9815

rate_2015_q1 = rate_2015_q2 / .9932

rate_2014_q4 = rate_2015_q1 / 1.0112

rate_2014_q3 = rate_2014_q4 / 1.0169

rate_2014_q2 = rate_2014_q3 / 1.0086

rate_2014_q1 = rate_2014_q2 / 1.0126

rate_2013_q4 = rate_2014_q1 / 0.9902

rate_2013_q3 = rate_2013_q4 / 1.0041

rate_2013_q2 = rate_2013_q3 / 1.0044

rate_2013_q1 = rate_2013_q2 / 1.0104

rate_2012_q4 = rate_2013_q1 / 0.9832

rate_2012_q3 = rate_2012_q4 / 1.0277

rate_2012_q2 = rate_2012_q3 / 1.0279

rate_2012_q1 = rate_2012_q2 / 1.0279

rate_2011_q4 = rate_2012_q1 / 1.076

rate_2011_q3 = rate_2011_q4 / 1.0236

rate_2011_q2 = rate_2011_q3 / 1

rate_2011_q1 = rate_2011_q2 / 1.011



# test data

test['average_q_price'] = 1



test_2016_q2_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month <= 7].index

test.loc[test_2016_q2_index, 'average_q_price'] = rate_2016_q2

# test.loc[test_2016_q2_index, 'year_q'] = '2016_q2'



test_2016_q1_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 1].loc[test['timestamp'].dt.month < 4].index

test.loc[test_2016_q1_index, 'average_q_price'] = rate_2016_q1

# test.loc[test_2016_q2_index, 'year_q'] = '2016_q1'



test_2015_q4_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 10].loc[test['timestamp'].dt.month < 12].index

test.loc[test_2015_q4_index, 'average_q_price'] = rate_2015_q4

# test.loc[test_2015_q4_index, 'year_q'] = '2015_q4'



test_2015_q3_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 7].loc[test['timestamp'].dt.month < 10].index

test.loc[test_2015_q3_index, 'average_q_price'] = rate_2015_q3

# test.loc[test_2015_q3_index, 'year_q'] = '2015_q3'



# test_2015_q2_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index

# test.loc[test_2015_q2_index, 'average_q_price'] = rate_2015_q2



# test_2015_q1_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index

# test.loc[test_2015_q1_index, 'average_q_price'] = rate_2015_q1





# train 2015

train['average_q_price'] = 1



train_2015_q4_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

# train.loc[train_2015_q4_index, 'price_doc'] = train.loc[train_2015_q4_index, 'price_doc'] * rate_2015_q4

train.loc[train_2015_q4_index, 'average_q_price'] = rate_2015_q4



train_2015_q3_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

#train.loc[train_2015_q3_index, 'price_doc'] = train.loc[train_2015_q3_index, 'price_doc'] * rate_2015_q3

train.loc[train_2015_q3_index, 'average_q_price'] = rate_2015_q3



train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

#train.loc[train_2015_q2_index, 'price_doc'] = train.loc[train_2015_q2_index, 'price_doc'] * rate_2015_q2

train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2



train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

#train.loc[train_2015_q1_index, 'price_doc'] = train.loc[train_2015_q1_index, 'price_doc'] * rate_2015_q1

train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1





# train 2014

train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

#train.loc[train_2014_q4_index, 'price_doc'] = train.loc[train_2014_q4_index, 'price_doc'] * rate_2014_q4

train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4



train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

#train.loc[train_2014_q3_index, 'price_doc'] = train.loc[train_2014_q3_index, 'price_doc'] * rate_2014_q3

train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3



train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

#train.loc[train_2014_q2_index, 'price_doc'] = train.loc[train_2014_q2_index, 'price_doc'] * rate_2014_q2

train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2



train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

#train.loc[train_2014_q1_index, 'price_doc'] = train.loc[train_2014_q1_index, 'price_doc'] * rate_2014_q1

train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1





# train 2013

train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

# train.loc[train_2013_q4_index, 'price_doc'] = train.loc[train_2013_q4_index, 'price_doc'] * rate_2013_q4

train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4



train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

# train.loc[train_2013_q3_index, 'price_doc'] = train.loc[train_2013_q3_index, 'price_doc'] * rate_2013_q3

train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3



train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

# train.loc[train_2013_q2_index, 'price_doc'] = train.loc[train_2013_q2_index, 'price_doc'] * rate_2013_q2

train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2



train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

# train.loc[train_2013_q1_index, 'price_doc'] = train.loc[train_2013_q1_index, 'price_doc'] * rate_2013_q1

train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1





# train 2012

train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

# train.loc[train_2012_q4_index, 'price_doc'] = train.loc[train_2012_q4_index, 'price_doc'] * rate_2012_q4

train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4



train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

# train.loc[train_2012_q3_index, 'price_doc'] = train.loc[train_2012_q3_index, 'price_doc'] * rate_2012_q3

train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3



train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

# train.loc[train_2012_q2_index, 'price_doc'] = train.loc[train_2012_q2_index, 'price_doc'] * rate_2012_q2

train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2



train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

# train.loc[train_2012_q1_index, 'price_doc'] = train.loc[train_2012_q1_index, 'price_doc'] * rate_2012_q1

train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1





# train 2011

train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

# train.loc[train_2011_q4_index, 'price_doc'] = train.loc[train_2011_q4_index, 'price_doc'] * rate_2011_q4

train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4



train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

# train.loc[train_2011_q3_index, 'price_doc'] = train.loc[train_2011_q3_index, 'price_doc'] * rate_2011_q3

train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3



train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

# train.loc[train_2011_q2_index, 'price_doc'] = train.loc[train_2011_q2_index, 'price_doc'] * rate_2011_q2

train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2



train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

# train.loc[train_2011_q1_index, 'price_doc'] = train.loc[train_2011_q1_index, 'price_doc'] * rate_2011_q1

train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1



train['price_doc'] = train['price_doc'] * train['average_q_price']

# train.drop('average_q_price', axis=1, inplace=True)



print('price changed done')



y_train = train["price_doc"]

# x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

# x_test = test.drop(["id", "timestamp"], axis=1)



x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)

x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)



num_train = len(x_train)

x_all = pd.concat([x_train, x_test])



for c in x_all.columns:

    if x_all[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_all[c].values))

        x_all[c] = lbl.transform(list(x_all[c].values))



x_train = x_all[:num_train]

x_test = x_all[num_train:]
print("x_train: ", x_train.shape)

print("y_train: ", y_train.shape)

print("x_test: ", x_test.shape)


def runLGBM(train_X, train_y, test_X, seed_val=50):

    params = {

        'boosting_type': 'gbdt',

        'objective': 'regression',

        'nthread': -1,

        'verbose': 0,

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.6,

#        'subsample_freq': 1,

        'colsample_bytree': 1, 

#        'reg_alpha': 1,

#        'reg_lambda': 0.001,

        'metric': 'rmse',

#        'min_split_gain': 0.5,

#        'min_child_weight': 1,

#        'min_child_samples': 10,

#        'scale_pos_weight': 1

    }

    pred_test_y = np.zeros(test_X.shape[0])

    train_set = lgbm.Dataset(train_X, train_y, silent=True)

    model = lgbm.train(params, train_set=train_set, num_boost_round=422)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model







#y_predict = model.predict(dtest)

y_predict, model = runLGBM(x_train, y_train, x_test, seed_val=50)



# y_predict = np.round(y_predict)

gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict, 'average_q_price': test['average_q_price']})

gunja_output['price_doc'] = gunja_output['price_doc'] * gunja_output['average_q_price']

gunja_output.drop('average_q_price', axis=1, inplace=True)

#gunja_output.head()

print("gunja_output.shape: ", gunja_output.shape)
print("x_train: ", x_train.shape)

print("y_train: ", y_train.shape)

print("x_test: ", x_test.shape)


train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

id_test = test.id



mult = .969



y_train = train["price_doc"] * mult + 10

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp"], axis=1)



for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_train[c].values))

        x_train[c] = lbl.transform(list(x_train[c].values))



for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_test[c].values))

        x_test[c] = lbl.transform(list(x_test[c].values))





def runLGBM_2(train_X, train_y, test_X, seed_val=50):

    params = {

        'boosting_type': 'gbdt',

        'objective': 'regression',

        'nthread': -1,

        'verbose': 0,

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.7,

#        'subsample_freq': 1,

        'colsample_bytree': 0.7, 

#        'reg_alpha': 1,

#        'reg_lambda': 0.001,

        'metric': 'rmse',

#        'min_split_gain': 0.5,

#        'min_child_weight': 1,

#        'min_child_samples': 10,

#        'scale_pos_weight': 1

    }

    pred_test_y = np.zeros(test_X.shape[0])

    train_set = lgbm.Dataset(train_X, train_y, silent=True)

    model = lgbm.train(params, train_set=train_set, num_boost_round=384)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model



y_predict, model = runLGBM_2(x_train, y_train, x_test, seed_val=50)

#y_predict = model.predict(dtest)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

# output.drop('average_q_price', axis=1, inplace=True)

# output.head()

print("output.shape: ", output.shape)


# Any results you write to the current directory are saved as output.

df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])





df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

y_train = df_train['price_doc'].values  * mult + 10









mult = 0.969

id_test = df_test['id']



df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

df_test.drop(['id'], axis=1, inplace=True)



num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

# Next line just adds a lot of NA columns (becuase "join" only works on indexes)

# but somewhow it seems to affect the result

df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')

print("df_all.shape:", df_all.shape)



# Add month-year

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df_all['month'] = df_all.timestamp.dt.month

df_all['dow'] = df_all.timestamp.dt.dayofweek



# Other feature engineering

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)

df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)



train['building_name'] = pd.factorize(train.sub_area + train['metro_km_avto'].astype(str))[0]

test['building_name'] = pd.factorize(test.sub_area + test['metro_km_avto'].astype(str))[0]



def add_time_features(col):

   col_month_year = pd.Series(pd.factorize(train[col].astype(str) + month_year.astype(str))[0])

   train[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())



   col_week_year = pd.Series(pd.factorize(train[col].astype(str) + week_year.astype(str))[0])

   train[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())



add_time_features('building_name')

add_time_features('sub_area')



def add_time_features(col):

   col_month_year = pd.Series(pd.factorize(test[col].astype(str) + month_year.astype(str))[0])

   test[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())



   col_week_year = pd.Series(pd.factorize(test[col].astype(str) + week_year.astype(str))[0])

   test[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())



add_time_features('building_name')

add_time_features('sub_area')





# Remove timestamp column (may overfit the model in train)

df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)





factorize = lambda t: pd.factorize(t[1])[0]



df_obj = df_all.select_dtypes(include=['object'])



X_all = np.c_[

    df_all.select_dtypes(exclude=['object']).values,

    np.array(list(map(factorize, df_obj.iteritems()))).T

]

print("X_all.shape: ", X_all.shape)



X_train = X_all[:num_train]

X_test = X_all[num_train:]





# Deal with categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()



for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]



df_values = pd.concat([df_numeric, df_obj], axis=1)



print("df_values.shape: ", df_values.shape)

# Convert to numpy values

X_all = df_values.values

print("X_all.shape: ", X_all.shape)



X_train = X_all[:num_train]

X_test = X_all[num_train:]



df_columns = df_values.columns





print(df_columns)
y_train
def runLGBM_3(train_X, train_y, test_X, seed_val=50):

    params = {

        'boosting_type': 'gbdt',

        'objective': 'regression',

        'nthread': -1,

        'verbose': 0,

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.7,

#        'subsample_freq': 1,

        'colsample_bytree': 0.7, 

#        'reg_alpha': 1,

#        'reg_lambda': 0.001,

        'metric': 'rmse',

#        'min_split_gain': 0.5,

#        'min_child_weight': 1,

#        'min_child_samples': 10,

#        'scale_pos_weight': 1

    }

    pred_test_y = np.zeros(test_X.shape[0])

    train_set = lgbm.Dataset(train_X, train_y, silent=True)

    model = lgbm.train(params, train_set=train_set, num_boost_round=384)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model



y_pred, model = runLGBM_3(x_train, y_train, x_test, seed_val=50)

#y_pred = model.predict(dtest)



df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})



print("df_sub.shape: ", df_sub.shape)

first_result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])

first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +

                                    .286*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2

result = first_result.merge(gunja_output, on="id", suffixes=['_follow','_gunja'])



result["price_doc"] = np.exp( .78*np.log(result.price_doc_follow) +

                              .22*np.log(result.price_doc_gunja) )

result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)

print("result.shape: ", result.shape)

result.to_csv('lightgbm_sub.csv', index=False)