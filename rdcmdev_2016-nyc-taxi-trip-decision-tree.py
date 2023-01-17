import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
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
kf = KFold(n_splits=3, shuffle=True, random_state=4305)
model = DecisionTreeRegressor()
param_grid = {
    'max_depth' : [11, 13, 15, 17, 21, 23]    
}
# https://www.kaggle.com/jpopham91/rmlse-vectorized
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.square(np.subtract(np.log1p(y_true), np.log1p(y_pred)))))
search_cv = GridSearchCV(model, param_grid, scoring=make_scorer(rmsle, greater_is_better=False),
                   cv=kf, verbose=3)
%%time
search_cv.fit(X, y)
results = pd.DataFrame(search_cv.cv_results_).sort_values(by='mean_test_score', ascending=False)
results.head()
search_cv.best_estimator_
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
y_pred = search_cv.best_estimator_.predict(X_test)
df_output = pd.DataFrame({'id' : X_id, 'trip_duration': y_pred})
df_output.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()