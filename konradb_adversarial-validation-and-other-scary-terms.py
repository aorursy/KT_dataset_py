import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from sklearn.metrics import roc_auc_score



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# We start by loading the training / test data and combining them with minimal preprocessing necessary

# Most of the data preparation is taken from here: 

# https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317

xtrain = pd.read_csv('../input/train.csv')

id_train = xtrain['id']

time_train = xtrain['timestamp']

ytrain = xtrain['price_doc']

xtrain.drop(['id', 'timestamp', 'price_doc'], axis = 1, inplace = True)

xtrain.fillna(-1, inplace = True)





xtest = pd.read_csv('../input/test.csv')

id_test = xtest['id']            

time_test = xtest['timestamp']

xtest.drop(['id', 'timestamp'], axis = 1, inplace = True)



# add identifier and combine

xtrain['istrain'] = 1

xtest['istrain'] = 0

xdat = pd.concat([xtrain, xtest], axis = 0)



# convert non-numerical columns to integers

df_numeric = xdat.select_dtypes(exclude=['object'])

df_obj = xdat.select_dtypes(include=['object']).copy()

    

for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]

    

xdat = pd.concat([df_numeric, df_obj], axis=1)

y = xdat['istrain']; xdat.drop('istrain', axis = 1, inplace = True)
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44)

xgb_params = {

        'learning_rate': 0.05, 'max_depth': 4,'subsample': 0.9,

        'colsample_bytree': 0.9,'objective': 'binary:logistic',

        'silent': 1, 'n_estimators':100, 'gamma':1,

        'min_child_weight':4

        }   

clf = xgb.XGBClassifier(**xgb_params, seed = 10)     
for train_index, test_index in skf.split(xdat, y):

        x0, x1 = xdat.iloc[train_index], xdat.iloc[test_index]

        y0, y1 = y.iloc[train_index], y.iloc[test_index]        

        print(x0.shape)

        clf.fit(x0, y0, eval_set=[(x1, y1)],

               eval_metric='logloss', verbose=False,early_stopping_rounds=10)

                

        prval = clf.predict_proba(x1)[:,1]

        print(roc_auc_score(y1,prval))