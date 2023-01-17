import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn import cross_validation





%matplotlib inline  

#read_data

train_data=pd.read_csv("../input/train.csv")

store_data=pd.read_csv("../input/store.csv")

test_data=pd.read_csv("../input/test.csv")

#dimensions of data

print("There are {} examples and {} features in training data".format(train_data.shape[0],train_data.shape[1]))

print("There are {} examples and {} features in store data".format(store_data.shape[0],store_data.shape[1]))

print("There are {} examples and {} features in test data".format(test_data.shape[0],test_data.shape[1]))
#Sampling dataset

train_data.sample()

store_data.sample()
test_data.sample()

percent_missing_train=train_data.isnull().sum()/train_data.isnull().count()

percent_missing_train
percent_missing_store=store_data.isnull().sum()/store_data.isnull().count()

percent_missing_store

percent_missing_test=test_data.isnull().sum()/test_data.isnull().count()

percent_missing_test

store_data[store_data[["Promo2SinceWeek","Promo2"]]["Promo2SinceWeek"].isnull()]
#making store data integral(storetype,Assortment)

store_data.loc[store_data['StoreType'] == 'a', 'StoreType'] = '1'

store_data.loc[store_data['StoreType'] == 'b', 'StoreType'] = '2'

store_data.loc[store_data['StoreType'] == 'c', 'StoreType'] = '3'

store_data.loc[store_data['StoreType'] == 'd', 'StoreType'] = '4'

store_data.loc[store_data['Assortment'] == 'a', 'Assortment'] = '1'

store_data.loc[store_data['Assortment'] == 'b', 'Assortment'] = '2'

store_data.loc[store_data['Assortment'] == 'c', 'Assortment'] = '3'

store_data
#store open if nan

test_data.fillna(1, inplace=True)

#only train with open stores

train_data = train_data[train_data["Open"] != 0]



#combining store table

train_data = pd.merge(train_data, store_data, on='Store')

test_data = pd.merge(test_data, store_data, on='Store')

#converting dates into year, month,date

train_data['year'] = train_data.Date.apply(lambda x: x.split('-')[0])

train_data['year'] = train_data['year'].astype(float)

train_data['month'] = train_data.Date.apply(lambda x: x.split('-')[1])

train_data['month'] = train_data['month'].astype(float)

train_data['day'] = train_data.Date.apply(lambda x: x.split('-')[2])

train_data['day'] = train_data['day'].astype(float)

test_data['year'] = test_data.Date.apply(lambda x: x.split('-')[0])

test_data['year'] = test_data['year'].astype(float)

test_data['month'] = test_data.Date.apply(lambda x: x.split('-')[1])

test_data['month'] = test_data['month'].astype(float)

test_data['day'] = test_data.Date.apply(lambda x: x.split('-')[2])

test_data['day'] = test_data['day'].astype(float)

train_data
test_data
#removing nans

test_data.fillna(0, inplace=True)

train_data.fillna(0, inplace=True)



train_data
test_data
#deleting id's

del test_data['Id']

del test_data['Date']

del test_data['PromoInterval']

del train_data['StateHoliday']





del train_data['Customers']

del train_data['Date']

del train_data['PromoInterval']

del test_data['StateHoliday']





train_data['StoreType'] = train_data['StoreType'].astype(float)

test_data['StoreType'] = test_data['StoreType'].astype(float)



train_data['Assortment'] = train_data['Assortment'].astype(float)

test_data['Assortment'] = test_data['Assortment'].astype(float)















train_features=list(test_data.columns)

train_features
test_data.columns
def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w





def rmspe(yhat, y):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe





def rmspe_xg(yhat, y):

    # y = y.values

    y = y.get_label()

    y = np.exp(y) - 1

    yhat = np.exp(yhat) - 1

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))

    return "rmspe", rmspe
params = {"objective": "reg:linear",

          "eta": 0.3,

          "max_depth": 8,

          "subsample": 0.7,

          "colsample_bytree": 0.7,

          "silent": 1

          }

num_trees = 300



print("Train a XGBoost model")

val_size = 100000

X_train, X_test = cross_validation.train_test_split(train_data, test_size=0.01)

dtrain = xgb.DMatrix(X_train[train_features], np.log(X_train["Sales"] + 1))

dvalid = xgb.DMatrix(X_test[train_features], np.log(X_test["Sales"] + 1))

dtest = xgb.DMatrix(test_data)

watchlist = [(dvalid, 'eval'), (dtrain, 'train')]

gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)



print("Validating")

train_probs = gbm.predict(xgb.DMatrix(X_test[train_features]))

indices = train_probs < 0

train_probs[indices] = 0

error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)

print('error', error)



print("Make predictions on the test set")

new_test_read=pd.read_csv("../input/test.csv")

test_probs = gbm.predict(xgb.DMatrix(test_data[train_features]))

indices = test_probs < 0

test_probs[indices] = 0

submission = pd.DataFrame({"Id": new_test_read["Id"], "Sales": np.exp(test_probs) - 1})

submission.to_csv("xgboost_kscript_submission.csv", index=False)
acc=1-error

print("The accuracy of model is {}".format(acc))