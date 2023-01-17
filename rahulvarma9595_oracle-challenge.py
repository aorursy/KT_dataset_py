# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

train = pd.read_csv('../input/dataset/train.csv')

train.head()



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/dataset/test.csv')

test
train_X = train.drop(['Global_active_power', 'Global_reactive_power',

       'Voltage'], axis=1)

train_Y = train['Global_active_power'].copy()
train_X['Date'] = [val.replace('/','-') for val in train_X['Date']]
train_X['Date/Time'] = train_X['Date'] + ' ' + train_X['Time']

train_X['Date/Time'] = pd.to_datetime(train_X['Date/Time'], dayfirst=True)
test['Date'] = [val.replace('/','-') for val in test['Date']]
test['Date/Time'] = test['Date'] + ' ' + test['Time']

test['Date/Time'] = pd.to_datetime(test['Date/Time'], dayfirst=True)
import datetime

dp = datetime.datetime.strptime('1970-01-11 00:00:00', '%Y-%m-%d %H:%M:%S')
train_X['seconds'] = [(val - pd.to_datetime(dp)).total_seconds()%31622400 for val in train_X['Date/Time']]
train_sec_min = train_X['seconds'].min()

train_sec_max = train_X['seconds'].max()
train_X['seconds'] = [((val - train_sec_min)/(train_sec_max - train_sec_min)) for val in train_X['seconds']]
test['seconds'] = [(val - pd.to_datetime(dp)).total_seconds()%31622400 for val in test['Date/Time']]
test_sec_min = test['seconds'].min()

test_sec_max = test['seconds'].max()
test['seconds'] = [((val - test_sec_min)/(test_sec_max - test_sec_min)) for val in test['seconds']]
train_X.drop(['Date','Time','Date/Time'],axis=1,inplace=True)

test_date = test['Date/Time'].copy()

test.drop(['Date','Time','Date/Time'],axis=1,inplace=True)
train_X['Global_intensity'] = pd.to_numeric(train_X['Global_intensity'], errors='coerce')

train_X['Sub_metering_1'] = pd.to_numeric(train_X['Sub_metering_1'], errors='coerce')

train_X['Sub_metering_2'] = pd.to_numeric(train_X['Sub_metering_2'], errors='coerce')

train_X['Sub_metering_3'] = pd.to_numeric(train_X['Sub_metering_3'], errors='coerce')
train_gi_min = train_X['Global_intensity'].min()

train_gi_max = train_X['Global_intensity'].max()

train_sm1_min = train_X['Sub_metering_1'].min()

train_sm1_max = train_X['Sub_metering_1'].max()

train_sm2_min = train_X['Sub_metering_2'].min()

train_sm2_max = train_X['Sub_metering_2'].max()

train_sm3_min = train_X['Sub_metering_3'].min()

train_sm3_max = train_X['Sub_metering_3'].max()
train_X['Global_intensity'] = [((val - train_gi_min)/(train_gi_max - train_gi_min)) for val in train_X['Global_intensity']]

train_X['Sub_metering_1'] = [((val - train_sm1_min)/(train_sm1_max - train_sm1_min)) for val in train_X['Sub_metering_1']]

train_X['Sub_metering_2'] = [((val - train_sm2_min)/(train_sm2_max - train_sm2_min)) for val in train_X['Sub_metering_2']]

train_X['Sub_metering_3'] = [((val - train_sm3_min)/(train_sm3_max - train_sm3_min)) for val in train_X['Sub_metering_3']]
test['Global_intensity'] = pd.to_numeric(test['Global_intensity'], errors='coerce')

test['Sub_metering_1'] = pd.to_numeric(test['Sub_metering_1'], errors='coerce')

test['Sub_metering_2'] = pd.to_numeric(test['Sub_metering_2'], errors='coerce')

test['Sub_metering_3'] = pd.to_numeric(test['Sub_metering_3'], errors='coerce')
test_gi_min = test['Global_intensity'].min()

test_gi_max = test['Global_intensity'].max()

test_sm1_min = test['Sub_metering_1'].min()

test_sm1_max = test['Sub_metering_1'].max()

test_sm2_min = test['Sub_metering_2'].min()

test_sm2_max = test['Sub_metering_2'].max()

test_sm3_min = test['Sub_metering_3'].min()

test_sm3_max = test['Sub_metering_3'].max()
test['Global_intensity'] = [((val - test_gi_min)/(test_gi_max - test_gi_min)) for val in test['Global_intensity']]

test['Sub_metering_1'] = [((val - test_sm1_min)/(test_sm1_max - test_sm1_min)) for val in test['Sub_metering_1']]

test['Sub_metering_2'] = [((val - test_sm2_min)/(test_sm2_max - test_sm2_min)) for val in test['Sub_metering_2']]

test['Sub_metering_3'] = [((val - test_sm3_min)/(test_sm3_max - test_sm3_min)) for val in test['Sub_metering_3']]
test_copy = test.copy()

train_X_copy = train_X.copy()
null_cols = [col for col in train_X_copy.columns if train_X_copy[col].isnull().any()]
for col in null_cols:

    train_X_copy[col] = train_X_copy.fillna(train_X_copy[col].mean(skipna = True))

    test_copy[col] = test_copy.fillna(test_copy[col].mean(skipna = True))
train_Y = pd.to_numeric(train_Y, errors='coerce')
train_Y = train_Y.fillna(train_Y.mean(skipna = True))
from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(train_X_copy, train_Y)
predictions = my_model.predict(test_copy)
predictions_Series = pd.Series(predictions)
test_date_hourly = pd.DataFrame(columns=['DateTime'])

i=0

for val in range(0,44640,60):

    test_date_hourly.loc[i] = test_date[val]

    i = i+1
output = pd.DataFrame({'DateTime': test_date,

                       'Average_global_active_power': predictions_Series})

avg_output = pd.DataFrame(columns = {'DateTime'})

avg_output = output.groupby([output.DateTime.dt.date,output.DateTime.dt.hour], as_index=False).mean()
test_result = test_date_hourly.join(avg_output)
test_result.to_csv('submission.csv', index=False)