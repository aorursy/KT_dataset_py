# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
building_df = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")



train = train.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")

train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"])

del weather_train
train.loc[(train['meter']==0) & (train['site_id']==0) & (train['timestamp']<'2016-05-21 00:00:00'), 'drop'] = True

train = train[train['drop']!=True]
def average_imputation(df, column_name):

    imputation = df.groupby(['timestamp'])[column_name].mean()

    

    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)

    del imputation

    return df
train = average_imputation(train, 'wind_speed')



beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 

          (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]



for item in beaufort:

    train.loc[(train['wind_speed']>=item[1]) & (train['wind_speed']<item[2]), 'beaufort_scale'] = item[0]
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["weekday"] = train["timestamp"].dt.weekday

train["hour"] = train["timestamp"].dt.hour

train["weekday"] = train['weekday'].astype(np.uint8)

train["hour"] = train['hour'].astype(np.uint8)

train["month"] = train["timestamp"].dt.month

train['year_built'] = train['year_built']-1900

train['square_feet'] = np.log(train['square_feet'])
train['group'] = train['month']

train['group'].replace((6, 7, 8), 21, inplace=True)

train['group'].replace((9, 10, 11), 22, inplace=True)

train['group'].replace((3, 4, 5), 23, inplace=True)

train['group'].replace((1, 2, 12), 24, inplace=True)

train['group'].replace((21), 1, inplace=True)

train['group'].replace((22), 2, inplace=True)

train['group'].replace((23), 3, inplace=True)

train['group'].replace((24), 4, inplace=True)
import gc

del train["timestamp"]

del train["drop"]

gc.collect()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train["primary_use"] = le.fit_transform(train["primary_use"])



categoricals = ['building_id',"primary_use", "hour", "weekday", "meter"]
drop_cols = ['site_id', "sea_level_pressure", "wind_speed", 'wind_direction', 'month', 'dew_temperature', "group"]



numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage", 'beaufort_scale', 'precip_depth_1_hr', "floor_count"]



feat_cols = categoricals + numericals
meter_df = [train[train['meter']==i] for i in range(0,4)]

del train
targets = [np.log1p(item["meter_reading"]) for item in meter_df]



for item in meter_df:

    del item["meter_reading"]

    for col in drop_cols:

        del item[col]
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from sklearn.metrics import mean_squared_error

import lightgbm as lgb



params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

             "num_leaves": 1280,

            "learning_rate": 0.05,

            "feature_fraction": 0.85,

            "reg_lambda": 2

            }



folds = 3

seed = 666



kf = KFold(n_splits=folds, shuffle=False, random_state=seed)

models0 = []

for train_index, val_index in kf.split(meter_df[0]):

    train_X = meter_df[0][feat_cols].iloc[train_index]

    val_X = meter_df[0][feat_cols].iloc[val_index]

    train_y = targets[0].iloc[train_index]

    val_y = targets[0].iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=10000,

                valid_sets=(lgb_train, lgb_eval),

               early_stopping_rounds=100,

               verbose_eval = 100)

    models0.append(gbm)
params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

             "num_leaves": 1280,

            "learning_rate": 0.05,

            "feature_fraction": 0.85,

            "reg_lambda": 2

            }



folds = 3

seed = 666



kf = KFold(n_splits=folds, shuffle=False, random_state=seed)

models1 = []

for train_index, val_index in kf.split(meter_df[1]):

    train_X = meter_df[1][feat_cols].iloc[train_index]

    val_X = meter_df[1][feat_cols].iloc[val_index]

    train_y = targets[1].iloc[train_index]

    val_y = targets[1].iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=1000,

                valid_sets=(lgb_train, lgb_eval),

               early_stopping_rounds=100,

               verbose_eval = 100)

    models1.append(gbm)
params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

             "num_leaves": 1280,

            "learning_rate": 0.05,

            "feature_fraction": 0.85,

            "reg_lambda": 2

            }



folds = 3

seed = 666



kf = KFold(n_splits=folds, shuffle=False, random_state=seed)

models2 = []

for train_index, val_index in kf.split(meter_df[2]):

    train_X = meter_df[2][feat_cols].iloc[train_index]

    val_X = meter_df[2][feat_cols].iloc[val_index]

    train_y = targets[2].iloc[train_index]

    val_y = targets[2].iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=10000,

                valid_sets=(lgb_train, lgb_eval),

               early_stopping_rounds=100,

               verbose_eval = 100)

    models2.append(gbm)
params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

             "num_leaves": 1280,

            "learning_rate": 0.05,

            "feature_fraction": 0.85,

            "reg_lambda": 2

            }



folds = 3

seed = 666



kf = KFold(n_splits=folds, shuffle=False, random_state=seed)

models3 = []

for train_index, val_index in kf.split(meter_df[3]):

    train_X = meter_df[3][feat_cols].iloc[train_index]

    val_X = meter_df[3][feat_cols].iloc[val_index]

    train_y = targets[3].iloc[train_index]

    val_y = targets[3].iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=10000,

                valid_sets=(lgb_train, lgb_eval),

               early_stopping_rounds=100,

               verbose_eval = 100)

    models3.append(gbm)
import gc

del meter_df, train_X, val_X, lgb_train, lgb_eval, train_y, val_y, targets

gc.collect()
#preparing test data

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

test = test.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")

del building_df

gc.collect()

test["primary_use"] = le.transform(test["primary_use"])



weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")

del weather_test
test = average_imputation(test, 'wind_speed')



for item in beaufort:

    test.loc[(test['wind_speed']>=item[1]) & (test['wind_speed']<item[2]), 'beaufort_scale'] = item[0]

    

test['beaufort_scale'] = test['beaufort_scale'].astype(np.uint8)



test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = test["timestamp"].dt.hour

test["weekday"] = test["timestamp"].dt.weekday

test["weekday"] = test['weekday'].astype(np.uint8)

test["hour"] = test['hour'].astype(np.uint8)

test["month"] = test["timestamp"].dt.month

test['year_built'] = test['year_built']-1900

test['square_feet'] = np.log(test['square_feet'])



test['group'] = test['month']

test['group'].replace((6, 7, 8), 21, inplace=True)

test['group'].replace((9, 10, 11), 22, inplace=True)

test['group'].replace((3, 4, 5), 23, inplace=True)

test['group'].replace((1, 2, 12), 24, inplace=True)

test['group'].replace((21), 1, inplace=True)

test['group'].replace((22), 2, inplace=True)

test['group'].replace((23), 3, inplace=True)

test['group'].replace((24), 4, inplace=True)

test = test.drop(["sea_level_pressure", "wind_direction", "timestamp", 'site_id', "wind_speed", 'month', 'dew_temperature', "group"], axis=1)
test_df = [test[test['meter']==i] for i in range(0,4)]

del test
row_ids = [list(item['row_id']) for item in test_df]
drop_list = ['row_id', 'meter']



for item in test_df:

    for col in drop_list:

        del item[col]
from tqdm import tqdm

i=0

res0=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test_df[0].shape[0]/50000)))):

    res0.append(sum(np.expm1([model.predict(test_df[0].iloc[i:i+step_size]) for model in models0])/folds))

    i+=step_size
i=0

res1=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test_df[1].shape[0]/50000)))):

    res1.append(sum(np.expm1([model.predict(test_df[1].iloc[i:i+step_size]) for model in models1])/folds))

    i+=step_size
i=0

res2=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test_df[2].shape[0]/50000)))):

    res2.append(sum(np.expm1([model.predict(test_df[2].iloc[i:i+step_size]) for model in models2])/folds))

    i+=step_size
i=0

res3=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test_df[3].shape[0]/50000)))):

    res3.append(sum(np.expm1([model.predict(test_df[3].iloc[i:i+step_size]) for model in models3])/folds))

    i+=step_size
res0 = np.concatenate(res0)

res1 = np.concatenate(res1)

res2 = np.concatenate(res2)

res3 = np.concatenate(res3)
del test_df
res0 = pd.DataFrame(data=res0,columns=['meter_reading'])

res1 = pd.DataFrame(data=res1,columns=['meter_reading'])

res2 = pd.DataFrame(data=res2,columns=['meter_reading'])

res3 = pd.DataFrame(data=res3,columns=['meter_reading'])
res0['row_id'] = row_ids[0]

res1['row_id'] = row_ids[1]

res2['row_id'] = row_ids[2]

res3['row_id'] = row_ids[3]
del row_ids
res = pd.concat([res0, res1, res2, res3])
del res0, res1, res2, res3
submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

submission = submission.merge(res, left_on='row_id', right_on='row_id', how='inner')

submission = submission[['row_id', 'meter_reading_y']]

submission.columns = ['row_id', 'meter_reading']

submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0

submission.to_csv('submission.csv', index=False)

submission