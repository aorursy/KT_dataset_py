import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import *



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import lightgbm as lgb

import xgboost as xgb
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

hist_transactions=pd.read_csv("../input/historical_transactions.csv")
# looking for dimensions of data

train.head()
test.head()
hist_transactions.head()
train.shape, test.shape, hist_transactions.shape
train.info()
test.info()
hist_transactions.info()
train.describe(include='all')
hist_transactions.describe(include='all')
#Checking for NA values in train 

train.isna().sum().plot(kind='barh')

for i, v in enumerate(train.isna().sum()):

    plt.text( v,i, str(v))

plt.title('missing values count')
# Distribution of cards used first-time 



train['first_active_month']=pd.to_datetime(train['first_active_month'])

count = train['first_active_month'].dt.date.value_counts()

count= count.sort_index()

plt.figure(figsize=(14,6))

sns.barplot(count.index, count.values)

plt.xticks(rotation='vertical')

plt.xlabel('First active month')

plt.ylabel('Number of cards')

plt.title("First active month count in train set")

plt.show()
# Checking for the distributions of features using violin plot



# feature 1

plt.figure(figsize=(8,4))

sns.violinplot(x="feature_1", y='target', data=train)

plt.xlabel('Feature 1')

plt.ylabel('target score')

plt.title("Feature 1 distribution")

plt.show()



# feature 2

plt.figure(figsize=(8,4))

sns.violinplot(x="feature_2", y='target', data=train)

plt.xlabel('Feature 2')

plt.ylabel('target score')

plt.title("Feature 2 distribution")

plt.show()

 

# feature 3

plt.figure(figsize=(8,4))

sns.violinplot(x="feature_3", y='target', data=train)

plt.xlabel('Feature 3')

plt.ylabel('target score')

plt.title("Feature 3 distribution")

plt.show()
Avg_month_lag= np.round(hist_transactions.groupby('card_id')['month_lag'].agg('mean').reset_index())

train= pd.merge(train, Avg_month_lag, on="card_id")
num_trans = hist_transactions.card_id.value_counts().reset_index()

num_trans.columns = ["card_id", "hist_transactions/card"]

train= pd.merge(train, num_trans, on="card_id")
plt.scatter('hist_transactions/card', 'target', data=train)

plt.xlabel('Number of hist_transactions/card')

plt.ylabel('target score')

plt.title('Number of hist_transactions/card  vs target score')
pur_amt = hist_transactions.groupby("card_id")

pur_amt = pur_amt["purchase_amount"].agg('mean').reset_index()

pur_amt.columns = ["card_id", "purchase_amt"]

train= pd.merge(train, pur_amt, on="card_id")
plt.scatter('hist_transactions/card', 'target', data=train)

plt.xlabel('Number of hist_transactions/card')

plt.ylabel('target score')

plt.title('Number of hist_transactions/card  Vs target score')
installments_percard = np.round(hist_transactions.groupby('card_id')['installments'].agg('mean').reset_index())

train= pd.merge(train, installments_percard, on="card_id")
plt.scatter('installments', 'target', data=train)

plt.xlabel('no.of installments')

plt.ylabel('target score')

plt.title('no.of installments Vs target scre')
train['first_active_month']=pd.to_datetime(train['first_active_month'])



train['first_active_year']=train['first_active_month'].dt.year

train['first_active_month']=train['first_active_month'].dt.month

plt.scatter(range(train.shape[0]), np.sort(train.target))

plt.ylabel('target Score')

plt.title('target-score distribution')
sns.heatmap(train.corr(), annot=True)

plt.title('Correlation map')
train_x=train.drop(['target',  'card_id'], axis=1)

train_y=train['target']
x_train, x_test, y_train, y_test=train_test_split(train_x, train_y, test_size=0.33)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
model=LinearRegression()

model.fit(x_train, y_train)
predict=model.predict(x_test)

predict_train=model.predict(x_train)
print('RMSE test:', np.sqrt(np.mean((predict - y_test)**2)))

print('RMSE train:', np.sqrt(np.mean((predict_train - y_train)**2)))
model_rf=RandomForestRegressor()

model_rf.fit(x_train, y_train)
predict_rf=model_rf.predict(x_test)

predict_rf_train=model_rf.predict(x_train)
print('Test RMSE RF:', np.sqrt(np.mean((predict_rf - y_test)**2)))

print('Train RMSE RF:', np.sqrt(np.mean((predict_rf_train - y_train)**2)))
Random_Search_Params ={

    'max_features':[1,2,3,4,5,6,7,8,9,10],

    "max_depth": list(range(1,train.shape[1])),

    'n_estimators' : [1, 2, 4, 8, 50, 100,150, 200, 250, 300],

    "min_samples_leaf": [5,10,15,20,25],

    'random_state' : [42] 

    }





random_search = RandomizedSearchCV(

    estimator=RandomForestRegressor(),

    param_distributions= Random_Search_Params, 

    cv=3,

    refit=True,

    verbose=True)
random_search.fit(x_train, y_train)
random_search.best_params_




model_rf_tune=RandomForestRegressor( random_state=42, 

                                     n_estimators=250, min_samples_leaf=15,

                                     max_features=6, max_depth=7 )
model_rf_tune.fit(x_train, y_train)
predict_rf_tune=model_rf_tune.predict(x_test)



predict_rf_tune_train=model_rf_tune.predict(x_train)
print('Test RMSE RF_tune_:', np.sqrt(np.mean((predict_rf_tune - y_test)**2)))

print('Train RMSE RF_tune:', np.sqrt(np.mean((predict_rf_tune_train - y_train)**2)))
params = {'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.01,

         "boosting": "gbrt",

         "metric": 'rmse'}



lgb_model = lgb.LGBMRegressor(**params, n_estimators = 10000,  n_jobs = -1)

lgb_model.fit(x_train, y_train, 

        eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=1000)
xgb_params = {'eta': 0.01,

              'objective': 'reg:linear',

              'max_depth': 6,

              'min_child_weight': 3,

              'subsample': 0.8,

              

              'eval_metric': 'rmse',

              'seed': 11,

              'silent': True}



model_xgb = xgb.XGBRegressor() 

model_xgb.fit(x_train, y_train)
trainPredict_xgb = model_xgb.predict(x_train)

testPredict_xgb = model_xgb.predict(x_test)



print("xgb test RMSE:", np.sqrt(mean_squared_error(y_test, testPredict_xgb)))

print("xgb train RMSE:", np.sqrt( mean_squared_error(y_train, trainPredict_xgb)))