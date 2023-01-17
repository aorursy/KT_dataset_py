import numpy as np

import pandas as pd 

import datetime as dt

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
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
train = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/train.zip')

test = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/test.zip')

sample_submission = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/sample_submission.zip')
train.head(3)
test.head(3)
train.info()
train.isnull().sum()
from scipy import stats

from scipy.stats import norm
plt.scatter(range(train.shape[0]),np.sort(train['trip_duration']))
sns.distplot(train.trip_duration.values, fit = norm)
sns.distplot(np.log1p(train.trip_duration.values), fit = norm)
train['trip_duration'] = np.log(train['trip_duration'].values)
feature_names=list(test)

df_train=train[feature_names]

df=pd.concat((df_train, test))
print(train.shape, test.shape, df.shape)
df.head(3)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['month'] = df['pickup_datetime'].dt.month

df['day'] = df['pickup_datetime'].dt.day

df['weekday'] = df['pickup_datetime'].dt.weekday

df['hour'] = df['pickup_datetime'].dt.hour

df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
df.drop(['pickup_datetime'], axis=1, inplace=True)
sns.countplot(df['hour'])
sns.countplot(df['dayofweek'])
df['dist_long'] = df['pickup_longitude'] - df['dropoff_longitude']

df['dist_lat'] = df['pickup_latitude'] - df['dropoff_latitude']
df['dist'] = np.sqrt(np.square(df['dist_long']) + np.square(df['dist_lat']))
def ft_haversine_distance(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371 #km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



df['distance'] = ft_haversine_distance(df['pickup_latitude'].values,

                                       df['pickup_longitude'].values, 

                                       df['dropoff_latitude'].values,

                                       df['dropoff_longitude'].values)
df.boxplot(column='distance')
#df = df[(df.distance < 200)]
g_vendor = train.groupby('vendor_id')['trip_duration'].mean()

sns.barplot(g_vendor.index,g_vendor.values)
sfflag = train.groupby('store_and_fwd_flag')['trip_duration'].mean()

sns.barplot(sfflag.index,sfflag.values)
pc = train.groupby('passenger_count')['trip_duration'].mean()

sns.barplot(pc.index,pc.values)
df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'],prefix = 'store')], axis=1)

df.drop(['store_and_fwd_flag'], axis=1, inplace=True)



df = pd.concat([df, pd.get_dummies(df['vendor_id'],prefix = 'vendor')], axis=1)

df.drop(['vendor_id'], axis=1, inplace=True)
df.head(3)
cor = df.corr()

mask = np.array(cor)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(cor,mask= mask,square=True,annot=True)
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_log_error

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.linear_model import LinearRegression
df.head(3)
df.drop(["id"], axis=1, inplace=True)
new_train = df[:train.shape[0]]

new_test = df[train.shape[0]:]
target = train['trip_duration']
X_train, X_val, y_train, y_val = train_test_split(new_train, target, test_size=0.2, shuffle=True)
def rmsle_score(preds, true):

    rmsle_score = (np.sum((np.log1p(preds)-np.log1p(true))**2)/len(true))**0.5

    return rmsle_score
from sklearn.metrics.scorer import make_scorer



RMSLE = make_scorer(rmsle_score)
import statsmodels.api as sm
model = sm.OLS(target.values, new_train.astype(float))
re = model.fit()

re.summary()
import lightgbm as lgbm
lgb_params = {

    'metric' : 'rmse',

    'learning_rate': 0.1,

    'max_depth': 25,

    'num_leaves': 1000, 

    'objective': 'regression',

    'feature_fraction': 0.9,

    'bagging_fraction': 0.5,

    'max_bin': 1000 }

lgb_df = lgbm.Dataset(new_train,target)
lgb_model = lgbm.train(lgb_params, lgb_df, num_boost_round=1500)
pred = lgb_model.predict(new_test)
pred_lgb = np.exp(pred)
import xgboost as xgb
params = {

    'booster':            'gbtree',

    'objective':          'reg:linear',

    'learning_rate':      0.1,

    'max_depth':          14,

    'subsample':          0.8,

    'colsample_bytree':   0.7,

    'colsample_bylevel':  0.7,

    'silent':             1

}
dtrain = xgb.DMatrix(new_train, target)
gbm = xgb.train(params,

                dtrain,

                num_boost_round = 200)
pred_test = np.exp(gbm.predict(xgb.DMatrix(new_test)))
#ensemble = (0.8*pred_lgb + 0.4*pred_test) 0.42295

#ensemble = (0.7*pred_lgb + 0.3*pred_test) 0.38148

ensemble = (0.6*pred_lgb + 0.4*pred_test) #0.38124

#ensemble = (0.55*pred_lgb + 0.45*pred_test) 0.38126
sub = pd.DataFrame()

sub['id'] = test.id

sub['trip_duration'] = ensemble

sub.head(3)
sub = pd.DataFrame()

sub['id'] = test.id

sub['trip_duration'] = ensemble

sub.head(3)
sub.to_csv('submission.csv', index=False)