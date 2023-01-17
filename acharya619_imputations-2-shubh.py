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

from matplotlib import pyplot as plt

from sklearn import metrics

data = pd.read_csv('/kaggle/input/rk-puram-ambient-air/rk.csv')

data['PM25'] = data['PM2.5']

data = data.drop('PM2.5', axis=1)

data = data.replace(0,float('nan'))

data.describe()
data = data[16091:]

data = data.drop('Toluene', axis=1)

data['datetime'] = pd.to_datetime(data['From Date'], format='%d/%m/%Y %H:%M')

data = data.drop(['From Date', 'To Date', 'VWS'], axis=1)

data = data.set_index('datetime')

data['Hour'] = data.index.hour

data['Year'] = data.index.year

data['Month'] = data.index.month

data['Weekday'] = data.index.weekday_name

data.isnull().sum()
print('AT: ', min(data['AT']), max(data['AT']))

print('BP: ', min(data['BP']), max(data['BP']))

print('RH: ', min(data['RH']), max(data['RH']))

print('SR: ', min(data['SR']), max(data['SR']))

print('WD: ', min(data['WD']), max(data['WD']))

print('WS: ', min(data['WS']), max(data['WS']))

print('CO: ', min(data['CO']), max(data['CO']))

print('NH3: ', min(data['NH3']), max(data['NH3']))

print('NO: ', min(data['NO']), max(data['NO']))

print('NO2: ', min(data['NO2']), max(data['NO2']))

print('NOx: ', min(data['NOx']), max(data['NOx']))

print('Ozone: ', min(data['Ozone']), max(data['Ozone']))

print('SO2: ', min(data['SO2']), max(data['SO2']))

print('PM2.5: ', min(data['PM25']), max(data['PM25']))

print('PM10: ', min(data['PM10']), max(data['PM10']))
data_Ozone = pd.DataFrame(data['Ozone'])

a = data_Ozone.Ozone.values  # Extract out relevant column from dataframe as array

m = np.concatenate(( [True], ~np.isnan(a), [True] ))  # Mask

ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits

start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits

print("start: %d, stop: %d" %(start,stop))

print(data_Ozone[start:stop-1].shape)

print(data_Ozone[start:stop-1].isnull().sum())

#maximum 81 samples are continuously missing in each column
data['AT'] = data.AT.interpolate(method='linear', limit_area='inside')

data['BP'] = data.BP.fillna(data.BP.rolling(83,min_periods=1,).median())

data ['RH'] = data.RH.interpolate(method='linear', limit_area='inside')

data ['SR'] = data.SR.interpolate(method='linear', limit_area='inside')

data ['WD'] = data.WD.interpolate(method='linear')

data ['WS'] = data.WS.interpolate(method='linear', limit_area='inside')

data ['CO'] = data.CO.interpolate(method='linear')

data ['NH3'] = data.NH3.interpolate(method='linear')

data ['NO'] = data.NO.interpolate(method='linear', limit_area='inside')

data ['NO2'] = data.NO2.interpolate(method='linear', limit_area='inside')

data ['NOx'] = data.NOx.interpolate(method='linear', limit_area='inside')

data ['Ozone'] = data.Ozone.interpolate(method='linear', limit_area='inside')

data ['PM10'] = data.PM10.interpolate(method='linear')

data ['PM25'] = data.PM25.interpolate(method='linear', limit_area='inside')

data ['SO2'] = data.SO2.fillna(data.SO2.rolling(83,min_periods=1,).median())

print('AT: ', min(data['AT']), max(data['AT']))

print('BP: ', min(data['BP']), max(data['BP']))

print('RH: ', min(data['RH']), max(data['RH']))

print('SR: ', min(data['SR']), max(data['SR']))#

print('WD: ', min(data['WD']), max(data['WD']))

print('WS: ', min(data['WS']), max(data['WS']))#

print('CO: ', min(data['CO']), max(data['CO']))

print('NH3: ', min(data['NH3']), max(data['NH3']))

print('NO: ', min(data['NO']), max(data['NO']))#

print('NO2: ', min(data['NO2']), max(data['NO2']))#

print('NOx: ', min(data['NOx']), max(data['NOx']))#

print('Ozone: ', min(data['Ozone']), max(data['Ozone']))#

print('SO2: ', min(data['SO2']), max(data['SO2']))

print('PM2.5: ', min(data['PM25']), max(data['PM25']))#

print('PM10: ', min(data['PM10']), max(data['PM10']))#
data.isnull().sum() #No. of missing values after imputing
data.to_csv('rk_imputed.csv', index=True)

#data['RH'] = data.RH.interpolate(method='akima', limit_area='inside')
data['AT']