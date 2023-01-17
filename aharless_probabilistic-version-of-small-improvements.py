# Parameters

prediction_stderr = 0.03  #  assumed standard error of predictions

                          #  (smaller values make output closer to input)

train_test_logmean_diff = 0.1  # assumed shift used to adjust frequencies for time trend

probthresh = 30  # minimum probability*frequency to use new price instead of just rounding

rounder = 2  # number of places left of decimal point to zero
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime

from scipy.stats import norm



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')

id_test = test.id
y_train = train["price_doc"]

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp"], axis=1)



for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_train[c].values)) 

        x_train[c] = lbl.transform(list(x_train[c].values))

        

for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_test[c].values)) 

        x_test[c] = lbl.transform(list(x_test[c].values))

        

xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)
num_boost_rounds = 384

model = xgb.train(xgb_params, dtrain, num_boost_round= num_boost_rounds)



y_predict = model.predict(dtest)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.head()
output.to_csv('before.csv', index=False)

preds = output
invest = train[train.product_type=="Investment"]

freqs = invest.price_doc.value_counts().sort_index()

print(freqs.head(20))

freqs.sample(10)
test_invest_ids = test[test.product_type=="Investment"]["id"]

invest_preds = pd.DataFrame(test_invest_ids).merge(preds, on="id")

invest_preds.head()
lnp = np.log(invest.price_doc)

stderr = lnp.std()

lfreqs = lnp.value_counts().sort_index()

lfreqs.head()
lnp_diff = train_test_logmean_diff

lnp_mean = lnp.mean()

lnp_newmean = lnp_mean + lnp_diff
def norm_diff(value):

    return norm.pdf((value-lnp_diff)/stderr) / norm.pdf(value/stderr)
newfreqs = lfreqs * (pd.Series(lfreqs.index.values-lnp_newmean).apply(norm_diff).values)



print( "What the middle of the adjusted and unadjusted freqs look like:")

print( lfreqs.values[880:900] )

print( newfreqs.values[880:900] )



print( "\nHeads")

print( lfreqs.head() )

print( newfreqs.head() )



print( "\nTails")

print( lfreqs.tail() )

print( newfreqs.tail() )



print( "\nSums")

print( lfreqs.sum() )

print( newfreqs.sum() )



print( "\nFirst prices that have nonzero frequencies:")

print( np.exp(newfreqs.index.values[0:20]) )



newfreqs.shape
stderr = prediction_stderr
lnpred = np.log(invest_preds.price_doc)

lnpred.head()
print(lnpred.shape)

print(newfreqs.index.values.shape)
mat =(np.array(newfreqs.index.values)[:,np.newaxis] - np.array(lnpred)[np.newaxis,:])/stderr

modelprobs = norm.pdf(mat)
freqprobs = pd.DataFrame( np.multiply( np.transpose(modelprobs), newfreqs.values ) )

freqprobs.index = invest_preds.price_doc.values

freqprobs.columns = freqs.index.values.tolist()

freqprobs.head()
prices = freqprobs.idxmax(axis=1)

priceprobs = freqprobs.max(axis=1)

mask = priceprobs<probthresh

prices[mask] = np.round(prices[mask].index,-rounder)
pr = invest_preds.price_doc

pd.DataFrame( {"id":test_invest_ids.values, "original":pr, "revised":prices.values}).head()
newpricedf = pd.DataFrame( {"id":test_invest_ids.values, "price_doc":prices} )

newpricedf.head()
preds.head()
newpreds = preds.merge(newpricedf, on="id", how="left", suffixes=("_old",""))

newpreds.loc[newpreds.price_doc.isnull(),"price_doc"] = newpreds.price_doc_old

newpreds.drop("price_doc_old",axis=1,inplace=True)

newpreds.head()
newpreds.to_csv('after.csv', index=False)