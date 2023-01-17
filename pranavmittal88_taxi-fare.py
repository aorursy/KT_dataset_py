# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows=10_000_000)
train_data.head()
train_data.shape
train_data.info()
test_data = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
test_data.head()
test_data.info()
train_data.isna().sum()
train_data['Difference_longitude']=np.abs(np.asarray(train_data['pickup_longitude']-train_data['dropoff_longitude']))
train_data['Difference_latitude']=np.abs(np.asarray(train_data['pickup_latitude']-train_data['dropoff_latitude']))


test_data['Difference_longitude']=np.abs(np.asarray(test_data['pickup_longitude']-test_data['dropoff_longitude']))
test_data['Difference_latitude']=np.abs(np.asarray(test_data['pickup_latitude']-test_data['dropoff_latitude']))
print(f'Before Dropping null values: {len(train_data)}')
train_data.dropna(inplace=True)
print(f'After Dropping null values: {len(train_data)}')
plot = train_data[:2000].plot.scatter('Difference_longitude', 'Difference_latitude')

train_data=train_data[(train_data['Difference_longitude']<5.0)&(train_data['Difference_latitude']<5.0)]
ls1 = list(train_data['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data['pickuptime'] = ls1

ls1 = list(test_data['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i] = ls1[i]