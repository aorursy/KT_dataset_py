import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
train_df = pd.read_csv('../input/train.csv', nrows = 100000)
train_df.shape
test_df = pd.read_csv('../input/test.csv')
test_df.shape
train_df.head(5)
train_df.isnull().sum()
train_df.dropna(inplace=True)
train_df.describe()
train_df = train_df[train_df['fare_amount']>0]
train_df.shape
def distance(lat1, lon1, lat2, lon2):
    a = 0.5 - np.cos((lat2 - lat1) *  0.017453292519943295)/2 + np.cos(lat1 * 0.017453292519943295) * np.cos(lat2 * 0.017453292519943295) * (1 - np.cos((lon2 - lon1) *  0.017453292519943295)) / 2
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res
train_df['distance'] = distance(train_df.pickup_latitude, train_df.pickup_longitude, \
                                      train_df.dropoff_latitude,train_df.dropoff_longitude)
test_df['distance'] = distance(test_df.pickup_latitude, test_df.pickup_longitude, \
                                      test_df.dropoff_latitude,test_df.dropoff_longitude)
train_df = train_df[train_df['distance']<15]
train_df.describe()
train_df = train_df[(train_df['passenger_count']!=0) & (train_df['passenger_count']<10)]
# train_df['hour'] = train_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).hour)
# train_df['year'] = train_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).year)

# test_df['hour'] = test_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).hour)
# test_df['year'] = test_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).year)
feat_cols_s = ['distance','passenger_count']

X = train_df[feat_cols_s]
y = train_df['fare_amount']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# from sklearn.ensemble import RandomForestRegressor
# r_reg= RandomForestRegressor(n_estimators=500)
# r_reg.fit(X_train,y_train)
# y_pred_final = r_reg.predict(test_df[feat_cols_s])

# submission = pd.DataFrame(
#     {'key': test_df.key, 'fare_amount': y_pred_final},
#     columns = ['key', 'fare_amount'])
# submission.to_csv('Random Forest regression.csv', index = False)
import xgboost as xgb
def XGBoost(X_train,X_test,y_train,y_test,num_rounds=500):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=dtrain,num_boost_round=num_rounds, 
                    early_stopping_rounds=20,evals=[(dtest,'test')],)
xgbm = XGBoost(X_train,X_test,y_train,y_test)
xgbm_pred = xgbm.predict(xgb.DMatrix(test_df[feat_cols_s]), ntree_limit = xgbm.best_ntree_limit)
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount':xgbm_pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('XGboost regression.csv', index = False)