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
def reduce_memory(df):

    for c in df.columns:

        if df[c].dtype=='int64':

            df[c] = df[c].astype('int32')

        elif df[c].dtype=='float64':

            df[c] = df[c].astype('float32')

    return df
test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')

building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
test = reduce_memory(test)

weather_test = reduce_memory(weather_test)

building = reduce_memory(building)
building['age'] = building['year_built'].max() - building['year_built'] + 1
convert = {'Education': 0,

 'Entertainment/public assembly': 1,

 'Food sales and service': 2,

 'Healthcare': 3,

 'Lodging/residential': 4,

 'Manufacturing/industrial': 5,

 'Office': 6,

 'Other': 7,

 'Parking': 8,

 'Public services': 9,

 'Religious worship': 10,

 'Retail': 11,

 'Services': 12,

 'Technology/science': 13,

 'Utility': 14,

 'Warehouse/storage': 15}
building['primary_use'] = building['primary_use'].map(convert)
test = pd.merge(test, building, on='building_id', how='left')

test = pd.merge(test, weather_test, on=['site_id','timestamp'], how='left')

test['timestamp'] = pd.to_datetime(test['timestamp'])

test['hour'] = test['timestamp'].dt.hour

test['weekday'] = test['timestamp'].dt.weekday



features = ['primary_use', 'square_feet', 'year_built', 'floor_count', 'age',

       'air_temperature', 'cloud_coverage', 'dew_temperature',

       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',

       'wind_speed', 'hour', 'weekday']

target = ['meter_reading']



for meter in [0,1,2,3]:

    x = test[test['meter']==meter][features]

    model = joblib.load(f'../input/ashrae-3/model_{meter}.model')

    pred = model.predict(x)

    del x



    pred = np.where(pred<0, 0, pred)

    test.loc[test['meter']==meter, 'meter_reading'] = pred
test[['row_id','meter_reading']].to_csv('submission.csv', index=False)