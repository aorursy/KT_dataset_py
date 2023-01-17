import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import model_selection, preprocessing, metrics

import xgboost as xgb

import datetime

#now = datetime.datetime.now()



otrain = pd.read_csv('../input/train.csv')



train = otrain.iloc[0:24000].copy()

test = otrain.iloc[24000:].copy()



id_test = test.id





y_train = train["price_doc"]

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp", "price_doc"], axis=1)



lbl1 = {}

lbl2 = {}



for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl1[c] = preprocessing.LabelEncoder()

        lbl1[c].fit(list(x_train[c].values)) 

        x_train[c] = lbl1[c].transform(list(x_train[c].values))

        #x_train.drop(c,axis=1,inplace=True)

        

for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl2[c] = preprocessing.LabelEncoder()

        lbl2[c].fit(list(x_test[c].values)) 

        x_test[c] = lbl2[c].transform(list(x_test[c].values))

        #x_test.drop(c,axis=1,inplace=True)        



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



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=True)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)



y_predict = model.predict(dtest)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

#output.head()



def rmsle(tgt, preds):

    return np.sqrt(sklearn.metrics.mean_squared_log_error(tgt, preds))

# oops, but i don't feel like rerunning that code ;)

import sklearn.metrics



# now split the held-back predicted output and run several different 35/65% splits on it...



sp = int(len(test) * .35)



for seeds in [0, 1337, 31337, 71331, 12345, 54321]:

    ind = output.index.copy()

    np.random.seed(seeds)

    ind = np.random.permutation(ind)

    

    tmp = test[['price_doc', 'id']].copy()

    tmp['preds'] = output.price_doc

    

    tmp = tmp.loc[ind]

    

    print(rmsle(tmp.iloc[0:sp].price_doc, tmp.iloc[0:sp].preds), rmsle(tmp.iloc[sp:].price_doc, tmp.iloc[sp:].preds))
# let's also take the entire HB area for comparison...

print(rmsle(tmp.price_doc, tmp.preds))