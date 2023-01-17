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
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"] = train["timestamp"].dt.hour

train["day"] = train["timestamp"].dt.day

train["year"] = train["timestamp"].dt.year

train["weekend"] = train["timestamp"].dt.weekday

train["month"] = train["timestamp"].dt.month

train['year_built'] = train['year_built']-1900

train['square_feet'] = np.log(train['square_feet'])



del train["timestamp"]
del train["year"]
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train["primary_use"] = le.fit_transform(train["primary_use"])



categoricals = ["site_id", "building_id", "primary_use", "hour", "day", "weekend", "month", "meter"]
drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]



numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",

              "dew_temperature"]



feat_cols = categoricals + numericals
target = np.log1p(train["meter_reading"])



del train["meter_reading"] 



train = train.drop(drop_cols, axis = 1)
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import lightgbm as lgb



params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

            'subsample': 0.2,

            'learning_rate': 0.1,

            'feature_fraction': 0.9,

            'bagging_fraction': 0.9,

            'alpha': 0.1, 

            'lambda': 0.1

            }



folds = 3

seed = 666



kf = KFold(n_splits = folds, shuffle = True, random_state = seed)

models = []

for train_index, val_index in kf.split(train):

    train_X = train[feat_cols].iloc[train_index]

    val_X = train[feat_cols].iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=300,

                valid_sets=(lgb_train, lgb_eval),

               early_stopping_rounds=20,

               verbose_eval = 100)

    models.append(gbm)
import gc

del train, train_X, val_X, lgb_train, lgb_eval, train_y, val_y, target

gc.collect()
#preparing test data

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

test = test.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")

del building_df

gc.collect()

test["primary_use"] = le.transform(test["primary_use"])



weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

weather_test = weather_test.drop(drop_cols, axis = 1)



test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")

del weather_test
test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = test["timestamp"].dt.hour.astype(np.uint8)

test["year"] = test["timestamp"].dt.year.astype(np.uint16)

test["day"] = test["timestamp"].dt.day.astype(np.uint8)

test["weekend"] = test["timestamp"].dt.weekday.astype(np.uint8)

test["month"] = test["timestamp"].dt.month.astype(np.uint8)

test['year_built'] = test['year_built']-1900

test['square_feet'] = np.log(test['square_feet'])



test = test[feat_cols]
from tqdm import tqdm

i=0

res=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):

    res.append(np.expm1(sum([model.predict(test.iloc[i:i+step_size]) for model in models])/folds))

    i+=step_size
res = np.concatenate(res)
submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

submission['meter_reading'] = res

submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0

submission.to_csv('submission.csv', index=False)

submission
submission['meter_reading'].mode()
from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "submission.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe





# create a link to download the dataframe

create_download_link(filename = 'submission.csv')



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 
