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
import seaborn as sns
import warnings
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
%matplotlib inline
%%time
df = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=10000000)
df.info()
df.head()
df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')
df.dropna(how='any',axis='rows',inplace=True)
df.info()
df.shape
features = df[['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 
               'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
price = df['fare_amount']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=0.3, random_state=42)
X_train.shape, X_test.shape
from sklearn.metrics import r2_score, mean_squared_error

def adjusted_r2_score(y_true, y_pred, X_test):
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    adjusted_r2 = 1 - (1-r2)*(len(y_true)-1)/(len(y_true) - X_test.shape[1]-1)
    return adjusted_r2
%%time

xgr = xgb.XGBRegressor(objective='reg:linear', n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
xgr.fit(X_train.drop(columns=['pickup_datetime']), y_train)

y_pred = xgr.predict(X_test.drop(columns=['pickup_datetime']))

rsq_baseline_xgb = r2_score(y_true=y_test, y_pred=y_pred)
adj_rsq_baseline_xgb = adjusted_r2_score(y_true=y_test, y_pred=y_pred, X_test=X_test)
rmse_baseline_xgb = mean_squared_error(y_true=y_test, y_pred=y_pred) ** 0.5
print('R-sq:', rsq_baseline_xgb)
print('Adj. R-sq:', adj_rsq_baseline_xgb)
print('RMSE:', rmse_baseline_xgb)
df.describe()
sns.kdeplot(df['fare_amount'].values,shade=True)
fig = plt.figure(figsize = (14, 5))
title = fig.suptitle("Distribution of trips across the US", fontsize=14)
ax1 = fig.add_subplot(1,2, 1)
p = sns.kdeplot((df[(df['pickup_latitude']>= 30) & (df['pickup_latitude'] <= 50)]['pickup_latitude'].values),
                shade=True,
                ax=ax1)
t= ax1.set_title("Distribution of latitude")

ax2 = fig.add_subplot(1,2, 2)
p = sns.kdeplot((df[(df['pickup_longitude']>= -125) & (df['pickup_longitude'] <= -65)]['pickup_longitude'].values),
                shade=True,
                ax=ax2)
t = ax2.set_title("Distribution of longitude")
sns.kdeplot(df['passenger_count'].values,shade=True)
lat_long = {
    'min_lat':30,
    'max_lat':50,    
    'min_long':-125,
    'max_long':-65, 
}
filter = (df['fare_amount'].between(0.01, 1000) 
                   & df['passenger_count'].between(1, 8)
                   & df['pickup_latitude'].between(lat_long['min_lat'], lat_long['max_lat'])
                   & df['dropoff_latitude'].between(lat_long['min_lat'], lat_long['max_lat']) 
                   & df['pickup_longitude'].between(lat_long['min_long'], lat_long['max_long'])
                   & df['dropoff_longitude'].between(lat_long['min_long'], lat_long['max_long']))

df = df[filter]

features = df[['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 
               'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
price = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=0.3, random_state=42)
X_train.shape, X_test.shape
X_train.head()
%%time

xgr = xgb.XGBRegressor(objective='reg:linear', n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
xgr.fit(X_train.drop(columns=['pickup_datetime']), y_train)

y_pred = xgr.predict(X_test.drop(columns=['pickup_datetime']))

rsq_baseline2_xgb = r2_score(y_true=y_test, y_pred=y_pred)
adj_rsq_baseline2_xgb = adjusted_r2_score(y_true=y_test, y_pred=y_pred, X_test=X_test)
rmse_baseline2_xgb = mean_squared_error(y_true=y_test, y_pred=y_pred) ** 0.5
print('R-sq:', rsq_baseline2_xgb)
print('Adj. R-sq:', adj_rsq_baseline2_xgb)
print('RMSE:', rmse_baseline2_xgb)
def manhattan(start_coord, end_coord):
    
    pickup_lat, pickup_long = start_coord
    dropoff_lat, dropoff_long = end_coord    
    distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    return distance
X_train['manhattan_dist'] = X_train.apply(lambda row: manhattan(start_coord=(row['pickup_latitude'], 
                                                                             row['pickup_longitude']),
                                                                end_coord=(row['dropoff_latitude'], 
                                                                           row['dropoff_longitude'])), axis=1)

X_test['manhattan_dist'] = X_test.apply(lambda row: manhattan(start_coord=(row['pickup_latitude'], 
                                                                             row['pickup_longitude']),
                                                                end_coord=(row['dropoff_latitude'], 
                                                                           row['dropoff_longitude'])), axis=1)
X_train.head()
%%time

xgr = xgb.XGBRegressor(objective='reg:linear', n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
xgr.fit(X_train.drop(columns=['pickup_datetime']), y_train)

y_pred = xgr.predict(X_test.drop(columns=['pickup_datetime']))

rsq_manhattan_xgb = r2_score(y_true=y_test, y_pred=y_pred)
adj_rsq_manhattan_xgb = adjusted_r2_score(y_true=y_test, y_pred=y_pred, X_test=X_test)
rmse_manhattan_xgb = mean_squared_error(y_true=y_test, y_pred=y_pred) ** 0.5
print('R-sq:', rsq_manhattan_xgb)
print('Adj. R-sq:', adj_rsq_manhattan_xgb)
print('RMSE:', rmse_manhattan_xgb)
