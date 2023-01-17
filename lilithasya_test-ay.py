#!/usr/bin/env python
import time
import timeit
import pandas as pd
import numpy as np
import xgboost as xgb
from numpy import double


def rmspe(preds, y):
    y = y.get_label()
    y = np.exp(y) - 1
    preds = np.exp(preds) - 1
    w = np.zeros(y.shape, dtype=double)
    ind = y != 0
    w[ind] = 1./double(y[ind]**2)
    rmspe = np.sqrt(np.mean(w * (double(y) - double(preds))**2))
    return "RMSPE", rmspe

def getConvDict(v):
    returnDict = dict()
    i = 1;
    for value in v:
        if(not returnDict.has_key(value)):
            returnDict[value] = i
            i = i+1
    return returnDict

path = "../input/"

print("start: " + time.strftime("%H:%M:%S"))
startTime = timeit.default_timer()

train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
store = pd.read_csv(path + "store.csv")

#consider only open shops
train = train[train["Open"] == 1]
train = train[train["Sales"] != 0]

train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

#NAs to zero
train.fillna(0, inplace = True)
test.fillna(0, inplace = True)

#Separating the date columns
train['year'] = train.Date.apply(lambda x: x.split('-')[0])
train['year'] = train['year'].astype(double)
train['month'] = train.Date.apply(lambda x: x.split('-')[1])
train['month'] = train['month'].astype(double)
train['day'] = train.Date.apply(lambda x: x.split('-')[2])
train['day'] = train['day'].astype(double)

train.drop('Date', inplace=True, axis = 1)
train.drop('StateHoliday', inplace=True, axis = 1)

test['year'] = test.Date.apply(lambda x: x.split('-')[0])
test['year'] = test['year'].astype(double)
test['month'] = test.Date.apply(lambda x: x.split('-')[1])
test['month'] = test['month'].astype(double)
test['day'] = test.Date.apply(lambda x: x.split('-')[2])
test['day'] = test['day'].astype(double)

test.drop('Date', inplace=True, axis = 1)
test.drop('StateHoliday', inplace=True, axis = 1)

#convert categorical attributes to numerical 
convDict = getConvDict(train["StoreType"])
train["StoreType"]= train.StoreType.apply(convDict.get)
test["StoreType"]= test.StoreType.apply(convDict.get)

train["Assortment"] = train.Assortment.apply(convDict.get) # same keys...
test["Assortment"] = test.Assortment.apply(convDict.get) # same keys...

convDict = getConvDict(train["PromoInterval"])
train["PromoInterval"] = train.PromoInterval.apply(convDict.get)
test["PromoInterval"] = test.PromoInterval.apply(convDict.get)

print("Training data:")
print(train.head(n=5))
print("Testing data:")
print(test.head(n=5))

features = ["Store","DayOfWeek","Open","Promo","SchoolHoliday","StoreType","Assortment","CompetitionDistance","CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2","Promo2SinceWeek","Promo2SinceYear","PromoInterval","month","year","day"]

print("Columns train: " + len(train.columns).__str__())
print("Columns test: " + len(test.columns).__str__())

#Split training data
print("train rows: " + len(train).__str__())

msk = np.random.rand(len(train)) < 0.92
train_d = train[msk]
eval_d = train[~msk]

print("train_d rows: " + len(train_d).__str__())
print("eval_d  rows: " + len(eval_d).__str__())

dtrain = xgb.DMatrix(train_d[features], np.log(train_d["Sales"] + 1))
dvalid = xgb.DMatrix(eval_d[features], np.log(eval_d["Sales"] + 1))

param = {"objective": "reg:linear",
          "booster": "gbtree",
          "eta": 0.026,
          "max_depth": 10,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }

watchlist = [(dvalid, 'val'), (dtrain, 'train')]

gbm = xgb.train(param, dtrain, num_boost_round=2200,  early_stopping_rounds=100, evals=watchlist, feval=rmspe) 

test_probs = gbm.predict(xgb.DMatrix(test[features]))
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
submitFileName = "submit" + time.strftime("%m_%d_%H:%M") + ".csv"
submission.to_csv(submitFileName, index=False)
print("end: " + time.strftime("%H:%M:%S"))
print("Time: " + time.strftime("%H:%M:%S", time.gmtime(timeit.default_timer() - startTime)))