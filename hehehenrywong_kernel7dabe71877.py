



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

import statsmodels.formula.api as smf

import os

import math

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

unimportant_cols = ['wind_direction', 'wind_speed', 'sea_level_pressure']

target = 'meter_reading'



def load_data(source='train', path='/kaggle/input/ashrae-energy-prediction'):

    ''' load and merge all tables '''

    assert source in ['train', 'test']



    building = pd.read_csv(f'{path}/building_metadata.csv', dtype={'building_id':np.uint16, 'site_id':np.uint8})

    weather  = pd.read_csv(f'{path}/weather_{source}.csv', parse_dates=['timestamp'],

                                                           dtype={'site_id':np.uint8, 'air_temperature':np.float16,

                                                                  'cloud_coverage':np.float16, 'dew_temperature':np.float16,

                                                                  'precip_depth_1_hr':np.float16},

                                                           usecols=lambda c: c not in unimportant_cols)

    df = pd.read_csv(f'{path}/{source}.csv', dtype={'building_id':np.uint16, 'meter':np.uint8}, parse_dates=['timestamp'])

    

    df = df.merge(building, on='building_id', how='left')

    df = df.merge(weather, on=['site_id', 'timestamp'], how='left')

    return df



# load and display some samples

train = load_data('train')

train['hour'] = np.uint8(train['timestamp'].dt.hour)

train['weekday'] = np.uint8(train['timestamp'].dt.weekday)

train['log_square_feet'] = np.float16(np.log(train['square_feet']))

train['log_meter_reading'] = np.float16(np.log(train['meter_reading']))

subset = train.sample(n = 2000000, replace = False)

train = subset.iloc[:-200000]

train = train.sort_values(by=['timestamp'])

train = train.reset_index()

train = train.drop(columns = 'index')

train.head(7)
train.loc[train.log_meter_reading < 0] = 0
data_ratios = train.count()/len(train)

train.loc[:, data_ratios < 1.0].mean()

train = train.fillna(train.loc[:, data_ratios < 1.0].mean())


formula = 'log_meter_reading ~ log_square_feet + building_id + meter + air_temperature + dew_temperature + floor_count + primary_use - 1 + year_built + hour + site_id + weekday + cloud_coverage + precip_depth_1_hr'

model = smf.glm(formula = formula, data=train, family=sm.families.Gaussian(),missing = 'drop')

result = model.fit()
test = subset.iloc[-200000:]

test = test.sort_values(by=['timestamp'])

test = test.reset_index()

test = test.drop(columns = 'index')
data_ratios = test.count()/len(test)

test.loc[:, data_ratios < 1.0].mean()
test = test.fillna(test.loc[:, data_ratios < 1.0].mean())
preds = result.predict(test)
from math import exp

from math import log

from math import sqrt

def RMSLE(a, b):

    assert(len(a) == len(b))

    sum = 0

    d = 0

    for i in range(len(a)):

        sum += ((log(exp(a[i]) + 1) - log(exp(b[i]) + 1)) ** 2)

        

    return sqrt((1 / (len(a))) * (sum))
from matplotlib.pyplot import hist



test.loc[test.log_meter_reading < 0] = 0

preds.loc[preds < 0] = 0

hist(test.log_meter_reading)

hist(preds)




RMSLE(preds,test.log_meter_reading)