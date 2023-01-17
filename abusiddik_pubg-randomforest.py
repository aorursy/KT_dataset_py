import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import Counter
from pprint import pprint
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
sample_submission = pd.read_csv('../input/sample_submission_V2.csv')
train.shape
test.shape
train.head()
test.head()
#Check for null values
train.isnull().any().values
train = train.dropna()
test.isnull().any().values
train.winPlacePerc.plot(kind='hist')
train[train.killStreaks > 7].shape
#82 players were killed more than 7 enimies in a short time
#players with more headshotkills
train.headshotKills.value_counts()
#Lets add some more features
train['total_dist'] = train['swimDistance'] + train['walkDistance'] + train['rideDistance']
test['total_dist'] = test['swimDistance'] + test['walkDistance'] + test['rideDistance']
train['kills_with_assist'] = train['kills'] + train['assists']
test['kills_with_assist'] = test['kills'] + test['assists']
print("Average distance travelled by player is ",train['total_dist'].mean())
train.DBNOs.value_counts().head(10).plot(kind='bar')
plt.scatter(train['rideDistance'],train['roadKills'])
train['headshot_over_kills'] = train['headshotKills'] / train['kills']
train['headshot_over_kills'].fillna(0, inplace=True)
test['headshot_over_kills'] = test['headshotKills'] / test['kills']
test['headshot_over_kills'].fillna(0, inplace=True)
train['headshot_over_kills'].value_counts().head(5)
train.head(2)
train = train.drop(['Id','groupId','matchId'],axis=1)
train.shape
matchtype = train.matchType.unique()

matchtype.__len__()
match_dict = {}
for i,each in enumerate(matchtype):
    match_dict[each] = i
match_dict
train.matchType = train.matchType.map(match_dict)

matchtype_test = test.matchType.unique()
match_dict_test = {}
for i,each in enumerate(matchtype_test):
    match_dict_test[each] = i
test.matchType = test.matchType.map(match_dict_test)
y = train['winPlacePerc']
X = train.drop(['winPlacePerc'],axis=1)
X.shape,y.shape
y[:2]
X[:2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#Lets Normalize the train data
sc_X = StandardScaler()
X_trainsc = sc_X.fit_transform(X_train)
X_testsc = sc_X.transform(X_test)

lr = LinearRegression()
lr.fit(X_trainsc, y_train)
y_pred = lr.predict(X_testsc)
y_pred[:10]
rmse = sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print("RMSE = >",rmse)
print("MSE = >",mse)
print("R Squared = >",r2)
res = pd.DataFrame()
res['Actual'] = y_test
res['Predicted'] = y_pred
res['Difference'] = abs(y_test-y_pred)
res.head(10)
#Decision Tree Regressor
dt = DecisionTreeRegressor()
dt.fit(X_trainsc,y_train)
y_pred_dt = dt.predict(X_testsc)
rmse = sqrt(mean_squared_error(y_test, y_pred_dt))
mse = mean_squared_error(y_test, y_pred_dt)
r2 = r2_score(y_test,y_pred_dt)
print("RMSE = >",rmse)
print("MSE = >",mse)
print("R Squared = >",r2)
dt = pd.DataFrame()
dt['Actual'] = y_test
dt['Predicted'] = y_pred_dt
dt['Difference'] = abs(y_test-y_pred_dt)
dt.head(10)
parameters = {
                'max_depth': 1,'min_data_in_leaf': 85,'feature_fraction': 0.80,'bagging_fraction':0.8,'boosting_type':'gbdt',
                'learning_rate': 0.1, 'num_leaves': 30,'subsample': 0.8,'lambda_l2': 4,'objective': 'regression_l2',
                'application':'regression','num_boost_round':5000,'zero_as_missing': True,
                'early_stopping_rounds':100,'metric': 'mae','seed': 2
             }
train_data = lgb.Dataset(X_trainsc, y_train, silent=False)
test_data = lgb.Dataset(X_testsc, y_test, silent=False)
model = lgb.train(parameters, train_set = train_data,verbose_eval=500, valid_sets=test_data)
test = test.drop(['Id','groupId','matchId'],axis=1)
#Lets check the prediction with x_testsc 
pred_lgb_samp_sc = model.predict(X_testsc, num_iteration = model.best_iteration)
lgb_res= pd.DataFrame()
lgb_res['Actual'] = y_test
lgb_res['Predicted_sc'] = pred_lgb_samp_sc
lgb_res['Difference'] = abs(y_test-pred_lgb_samp_sc)
lgb_res.head(10)
# We'll normalize the test data aswell for better prediction

sc_test = StandardScaler()
test_sc = sc_test.fit_transform(test)
# prediction
pred_lgb_sc = model.predict(test_sc, num_iteration = model.best_iteration)
pred_lgb_sc[:10]
# Replace the prediction which is greator than 1 by 1 and less than 0 by 0

pred_lgb_sc[pred_lgb_sc > 1] = 1
pred_lgb_sc[pred_lgb_sc < 0] = 0
pred_lgb_sc.__len__()
sample_submission['winPlacePerc'] = pred_lgb_sc
sample_submission.to_csv('sample_submission.csv',index=False)


