import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from bayes_opt import BayesianOptimization

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

taxidata = pd.read_csv("../input/train.csv",nrows = 5_000_000) #Loading only a small subset of training data
taxifare_test = pd.read_csv("../input/test.csv")
taxidata.head()
taxidata.info()
print(taxidata.isnull().sum())
print(sum(taxidata['passenger_count']>9))
print(sum(taxidata['fare_amount']>1000))
plt.hist(taxidata['passenger_count'][:500000],bins=20)
timedate = pd.to_datetime(taxidata['pickup_datetime'])
taxidata['hour'] = timedate.apply(lambda x:x.hour)
taxidata['week']=timedate.apply(lambda x:x.weekofyear)
print(taxidata['hour'].nunique())
print(taxidata['week'].nunique())
taxidata['lat_diff'] = (taxidata['dropoff_latitude']-taxidata['pickup_latitude']).abs()
taxidata['lon_diff'] = (taxidata['dropoff_longitude']-taxidata['pickup_longitude']).abs()
print(sum((taxidata['lon_diff']>5) & (taxidata['lat_diff']>5)))
plt.scatter(taxidata['lat_diff'][:20000],taxidata['lon_diff'][:20000])
taxidata = taxidata.dropna()
taxidata = taxidata[(taxidata['lat_diff'] < 5.0) & (taxidata['lon_diff'] < 5.0)]
taxidata = taxidata[taxidata['passenger_count']<10]
taxidata = taxidata[taxidata['fare_amount']<1000]
taxidata_X = taxidata[['lat_diff','lon_diff','passenger_count','hour','week']]
taxidata_y = taxidata['fare_amount']
X_train, X_test_val, y_train, y_test_val = train_test_split(taxidata_X, taxidata_y, test_size=0.4)
#lm = LinearRegression()
#lm.fit(X_train,y_train)
#print(lm.intercept_)
#print(lm.coef_)
#test_val_pred = lm.predict(X_test_val)
#sns.distplot(y_test_val-test_val_pred)
#plt.scatter(y_test_val,test_val_pred)
#np.sqrt(metrics.mean_squared_error(y_test_val,test_val_pred))
#svr = svm.LinearSVR()
#svr.fit(X_train,y_train)
#test_val_pred_svr = svr.predict(X_test_val)
#np.sqrt(metrics.mean_squared_error(y_test_val,test_val_pred_svr))
#data_dmatrix = xgb.DMatrix(data=taxidata_X,label=taxidata_y)
#params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}
#cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
#                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
#cv_results.head()
#print((cv_results["test-rmse-mean"]).tail(1))
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest_val = xgb.DMatrix(X_test_val)

#Thanks to https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
def xgb_crossval(max_depth, gamma, colsample_bytree):
    params = {'eval_metric': 'rmse','max_depth': int(max_depth),'subsample': 0.8,'eta': 0.1,'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    cv = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv['test-rmse-mean'].iloc[-1]
from bayes_opt import BayesianOptimization
xgb_bayes = BayesianOptimization(xgb_crossval, {'max_depth': (3, 7), 'gamma': (0, 1),'colsample_bytree': (0.3, 0.9)})
xgb_bayes.maximize(init_points=3, n_iter=5, acq='ei')
params = xgb_bayes.max['params']
params['max_depth'] = int(params['max_depth'])

### Train a new model with the best parameters from the search
xgb_model = xgb.train(params, dtrain, num_boost_round=250)

# Predict on testing (validation) and training set
y_pred_val = xgb_model.predict(dtest_val)
y_train_pred = xgb_model.predict(dtrain)


# Testing and training RMSE
print(np.sqrt(mean_squared_error(y_test_val, y_pred_val)))
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
xgb.plot_importance(xgb_model)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
taxifare_test['lat_diff'] = (taxifare_test['dropoff_latitude']-taxifare_test['pickup_latitude']).abs()
taxifare_test['lon_diff'] = (taxifare_test['dropoff_longitude']-taxifare_test['pickup_longitude']).abs()
testtimedate = pd.to_datetime(taxifare_test['pickup_datetime'])
taxifare_test['hour']=testtimedate.apply(lambda x:x.hour)
taxifare_test['week']=testtimedate.apply(lambda x:x.weekofyear)
test_X = taxifare_test[['lat_diff','lon_diff','passenger_count','hour','week']]
#predictions_lm = lm.predict(test_X)
#predictions_svr = svr.predict(test_X)
dtest = xgb.DMatrix(test_X)
predictions_xgb = xgb_model.predict(dtest)
predictions = predictions_xgb
submission = pd.DataFrame(
    {'key': taxifare_test.key, 'fare_amount': predictions},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))
