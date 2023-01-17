# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import radians, cos, sin, asin, sqrt

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df =  pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 10_000_000)

train_df.dtypes
def haversine_np(lon1, lat1, lon2, lat2):

    "https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas"

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1

    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km
train_df.describe()
train_df['distance'] = haversine_np(train_df['pickup_longitude'], train_df['pickup_latitude'], train_df['dropoff_longitude'], train_df['dropoff_latitude'])
print('Old size: %d' % len(train_df))

train_df = train_df.dropna(how = 'any', axis = 'rows')

print('New size: %d' % len(train_df))
X = train_df[['distance']].values

Y = train_df['fare_amount'].values
kwargs = {'bootstrap': True,

 'max_depth': None,

 'min_samples_leaf': 31,

 'min_samples_split': 93}

rand_regr = RandomForestRegressor(n_estimators=3, **kwargs)
rand_regr.fit(X, Y)

rand_regr.score(X,Y)