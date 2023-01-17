import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn import preprocessing

from math import radians, cos, sin, asin, sqrt

from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.display.float_format = "{:.18f}".format
!ls ../input
df = pd.read_csv('../input/train.csv')
df.head()
df.dropna(inplace=True)
m = np.mean(df['trip_duration'])
s = np.std(df['trip_duration'])
df = df[df['trip_duration'] <= m + 2*s]
df = df[df['trip_duration'] >= m - 2*s]
plg, plt = 'pickup_longitude', 'pickup_latitude'
dlg, dlt = 'dropoff_longitude', 'dropoff_latitude'
pdt, ddt = 'pickup_datetime', 'dropoff_datetime'
# https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def euclidian_distance(x):
    x1, y1 = np.float64(x[plg]), np.float64(x[plt])
    x2, y2 = np.float64(x[dlg]), np.float64(x[dlt])    
    return haversine(x1, y1, x2, y2)
%%time
df['distance'] = df[[plg, plt, dlg, dlt]].apply(euclidian_distance, axis=1)
df.head()
df[pdt] = df[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df[ddt] = df[ddt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df['month'] = df[pdt].apply(lambda x : x.month)
df['weekDay'] = df[pdt].apply(lambda x : x.weekday())
df['dayMonth'] = df[pdt].apply(lambda x : x.day)
df['pickupTimeMinutes'] = df[pdt].apply(lambda x : x.hour * 60.0 + x.minute)
df.head()
df.drop(['id', pdt, ddt, dlg, dlt, 'store_and_fwd_flag'], inplace=True, axis=1)
df.head()
df = df[[plg, plt, 'distance', 'month', 'dayMonth', 'weekDay', 'pickupTimeMinutes', 'passenger_count', 'vendor_id', 'trip_duration']]
df.head()
X, y = df.iloc[:, :-1], df.iloc[:, -1]
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
dtrain = xgb.DMatrix(X, label=y)
xgb_pars = {
    'min_child_weight': 1, 
    'eta': 0.5, 
    'colsample_bytree': 0.9, 
    'max_depth': 6,
    'subsample': 0.9, 
    'lambda': 1., 
    'nthread': -1, 
    'booster' : 'gbtree', 
    'silent': 1,
    'eval_metric': 'rmse',
    'objective': 'reg:linear'
}
%%time
model = xgb.train(xgb_pars, dtrain, 10, maximize=False, verbose_eval=1)
xgb.plot_importance(model, max_num_features=28, height=0.7)
df_test = pd.read_csv('../input/test.csv')
df_test.head()
df_test['distance'] = df_test[[plg, plt, dlg, dlt]].apply(euclidian_distance, axis=1)
df_test[pdt] = df_test[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df_test['month'] = df_test[pdt].apply(lambda x : x.month)
df_test['weekDay'] = df_test[pdt].apply(lambda x : x.weekday())
df_test['dayMonth'] = df_test[pdt].apply(lambda x : x.day)
df_test['pickupTimeMinutes'] = df_test[pdt].apply(lambda x : x.hour * 60.0 + x.minute)
df_test.drop(['pickup_datetime', dlg, dlt, 'store_and_fwd_flag'], inplace=True, axis=1)
df_test = df_test[['id', plg, plt, 'distance', 'month', 'dayMonth', 'weekDay', 'pickupTimeMinutes', 'passenger_count', 'vendor_id']]
df_test.head()
X_id, X_test = df_test.iloc[:, 0], df_test.iloc[:, 1:]
X_id.shape, X_test.shape
scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)
dtest = xgb.DMatrix(X_test, label=y)
y_pred = model.predict(dtest)
df_output = pd.DataFrame({'id' : X_id})
df_output['trip_duration'] = pd.DataFrame(y_pred)
df_output.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()