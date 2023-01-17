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
import pandas as pd

import numpy as np

import seaborn as sns

import datetime

import numpy as np

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm
train_data = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

building_data= pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
def reduce_mem_usage(data, use_float16=False) -> pd.DataFrame:

    start_mem = data.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in data.columns:

#         if datetime.date(data[col]) or is_categorical_dtype(data[col]):

#             continue

        col_type = data[col].dtype



        if col_type != object:

            c_min = data[col].min()

            c_max = data[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    data[col] = data[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    data[col] = data[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    data[col] = data[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    data[col] = data[col].astype(np.int64)

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    data[col] = data[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    data[col] = data[col].astype(np.float32)

                else:

                    data[col] = data[col].astype(np.float64)

        else:

            data[col] = data[col].astype('category')



    end_mem = data.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.2f}%'.format(

        100 * (start_mem - end_mem) / start_mem))



    return data
train_data = train_data.merge(building_data, on="building_id", how="left")

train_data = train_data.merge(weather_train, on=["site_id", "timestamp"], how="left")
train_data.isnull().sum()
train_data = reduce_mem_usage(train_data, use_float16=True)
# train_data['floor_count'] = train_data['floor_count'].fillna(-999).astype(np.int16)

# train_data['year_built'] = train_data['year_built'].fillna(-999).astype(np.int16)

# train_data['cloud_coverage'] = train_data['cloud_coverage'].fillna(-999).astype(np.int16)
train_data['timestamp']= pd.to_datetime(train_data['timestamp'], format="%Y-%m-%d %H:%M:%S")

train_data["hour"] = train_data["timestamp"].dt.hour

train_data["weekday"] = train_data["timestamp"].dt.weekday

train_data["weekday"] = train_data['weekday'].astype(np.uint8)

train_data["hour"] = train_data['hour'].astype(np.uint8)

train_data['square_feet'] = np.log(train_data['square_feet'])

train_data['year_built'] = train_data['year_built']-1900
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train_data["primary_use"] = le.fit_transform(train_data["primary_use"])



categoricals = ["site_id", "building_id", "primary_use", "meter",  "wind_direction"]
numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage", "dew_temperature", 'floor_count']



feat_cols = categoricals + numericals
drop_cols = ["sea_level_pressure", "wind_speed"]



target = np.log1p(train_data["meter_reading"])



del train_data["meter_reading"]

del train_data["precip_depth_1_hr"]



train_data = train_data.drop(drop_cols, axis = 1)
train_data.head()
params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

            'subsample': 0.25,

            'subsample_freq': 1,

            'learning_rate': 0.4,

            'num_leaves': 20,

            'feature_fraction': 0.9,

            'lambda_l1': 1,  

            'lambda_l2': 1

            }



folds = 4

seed = 666



kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)



models = []

for train_index, val_index in kf.split(train_data, train_data['building_id']):

    train_X = train_data[feat_cols].iloc[train_index]

    val_X = train_data[feat_cols].iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=(lgb_train, lgb_eval),

                early_stopping_rounds=100,

                verbose_eval = 100)

    models.append(gbm)
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

test = test.merge(building_data, left_on = "building_id", right_on = "building_id", how = "left")

test["primary_use"] = le.transform(test["primary_use"])



weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")



test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
test.isnull().sum()
test = reduce_mem_usage(test, use_float16=True)
test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = test["timestamp"].dt.hour

test["weekday"] = test["timestamp"].dt.weekday

test["weekday"] = test['weekday'].astype(np.uint8)

test["hour"] = test['hour'].astype(np.uint8)

test['year_built'] = test['year_built']-1900

test['square_feet'] = np.log(test['square_feet'])

test["meter"] = test['meter'].astype(np.uint8)

test["site_id"] = test['site_id'].astype(np.uint8)



del test["precip_depth_1_hr"]



test = test[feat_cols]
test.head()
i=0

res=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):

    res.append(np.expm1(sum([model.predict(test.iloc[i:i+step_size]) for model in models])/folds))

    i+=step_size
res = np.concatenate(res)
len(res)


submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

submission['meter_reading'] = res

submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0

submission.to_csv('submission.csv', index=False)

submission