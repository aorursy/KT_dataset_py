

%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from sklearn import metrics

import gc

import joblib

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')
building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
train.head()
def reduce_memory(df):

    for c in df.columns:

        if df[c].dtype=='int64':

            df[c] = df[c].astype('int32')

        elif df[c].dtype=='float64':

            df[c] = df[c].astype('float32')

    return df
train = reduce_memory(train)

test = reduce_memory(test)

weather_train = reduce_memory(weather_train)

weather_test = reduce_memory(weather_test)
train.info()
test.info()
train['timestamp'].min(), train['timestamp'].max(), test['timestamp'].min(), test['timestamp'].max()
building.info()
building['primary_use'].value_counts(1)
building['year_built'].value_counts(1,dropna=False)
building.describe()
weather_train.head()
weather_train.info()
weather_train.describe()
weather_train['timestamp'].min(), weather_train['timestamp'].max(),weather_test['timestamp'].min(), weather_test['timestamp'].max()
building.head()
building['age'] = building['year_built'].max() - building['year_built'] + 1
lb = LabelEncoder()

building['primary_use'] = lb.fit_transform(building['primary_use'])
train.columns, building.columns, weather_train.columns
train = pd.merge(train, building, on='building_id', how='left')

train.columns
train.head()
weather_train.head()
train = pd.merge(train, weather_train, on=['site_id','timestamp'], how='left')
train['timestamp'] = pd.to_datetime(train['timestamp'])
train['hour'] = train['timestamp'].dt.hour

train['weekday'] = train['timestamp'].dt.weekday
train.columns
del train['building_id'], train['site_id']
gc.collect()
features = ['primary_use', 'square_feet', 'year_built', 'floor_count', 'age',

       'air_temperature', 'cloud_coverage', 'dew_temperature',

       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',

       'wind_speed', 'hour', 'weekday']

target = ['meter_reading']
train['meter'].unique()


    

def train_model(meter):

    print('start train model', meter)

    print('features', features)

    x = train[train['meter']==meter][features]

    y = train[train['meter']==meter][target]



    split = int(0.8*len(x))

    train_x = x.iloc[:split]

    val_x = x.iloc[split:]



    train_y = y.iloc[:split]

    val_y = y.iloc[split:]

    print('train', train_x.shape, 'val', val_x.shape)



    del x, y

    gc.collect()

    val_set = (val_x, val_y)

    model = lgb.LGBMRegressor(objective='regression', learning_rate=0.01, early_stopping_rounds=20)

    model.fit(train_x, train_y, eval_set=val_set, categorical_feature=['primary_use'], verbose=10)

    

    pred = model.predict(val_x)

    print('train_y')

    print(train_y.quantile([.25,.5,.75,.9]))

    print('pred')

    print(pd.Series(pred).quantile([.25,.5,.75,.9]))

    del train_x, train_y

    pred = np.where(pred<0, 0, pred)

    score = metrics.mean_squared_log_error(val_y, pred)

    print('meter:', meter, 'score:', score)

    joblib.dump(model, f'model_{meter}.model')

    return model
for m in [0,1,2,3]:

    model = train_model(m)
model