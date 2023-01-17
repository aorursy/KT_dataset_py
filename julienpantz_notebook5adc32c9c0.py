# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import subprocess

from scipy.sparse import csr_matrix, hstack

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import KFold

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.externals import joblib

from sklearn.metrics import fbeta_score, make_scorer

from sklearn import tree

from sklearn.cross_validation import cross_val_score

import time

from numpy.random import RandomState

prng = RandomState(1234567890)

import sklearn ; print(sklearn.__version__)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
test['loss'] = np.nan #Add a Loss column with NaN to test DF

joined = pd.concat([train, test])



for column in list(train.select_dtypes(include=['object']).columns):

    # 

    if train[column].nunique() != test[column].nunique(): # could fail in theory

        set_train = set(train[column].unique()) # Do we need unique since we have set

        set_test = set(test[column].unique())

        remove_train = set_train - set_test

        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)

        def filter_cat(x):

            if x in remove:

                return np.nan

            return x



        joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)

        

    #pd.factorize encode the (sorted) values

    joined[column] = pd.factorize(joined[column].values, sort=True)[0]

    

#HACK We set test['loss'] = NaN to recognize it 

train = joined[joined['loss'].notnull()]

test = joined[joined['loss'].isnull()]

y = train['loss']

train = train.drop(['id','loss'],1)

test = test.drop(['id','loss'],1)
def transform(y,is_inverse=None):

    shift = 200

    if is_inverse:

        return np.exp(y)-shift

    return np.log(y+shift)



def scorer(x,y):

    return mean_absolute_error(transform(x,True),transform(y,True))



custom_scorer = make_scorer(scorer,greater_is_better=False)
print('mean',np.mean(y),'median',np.median(y))

#print(train.head(5))



X = train.values

X_test = test.values

fy = transform(y)

print('mean',transform(np.mean(fy),True),'median',transform(np.median(fy),True))
if False:

    t0 = time.time()

    reg = tree.DecisionTreeRegressor(max_depth=9,min_samples_split=2)

    scores = cross_val_score(reg, X,fy, cv=3,scoring=custom_scorer)

    d = time.time()-t0

    print(scores.mean(),d)



    #reg = tree.DecisionTreeRegressor()    

    #reg = reg.fit(x,fy)
if False:

    params = [5,10,25,50,100]

    params = [5]

    for param in params:

        t0 = time.time()

        forest = RandomForestRegressor(n_estimators = param,criterion='mse',n_jobs=-1,random_state=prng)

        scores = cross_val_score(forest, X,fy, cv=3,scoring=custom_scorer)

        d = time.time()-t0

        print(param,scores.mean(),d)



    #5 -1319.6123701 37.71500015258789

    #10 -1266.15301356 102.43400001525879

    #25 -1231.19439691 213.24000000953674

    #50 -1219.07043097 292.1300001144409

    #100 -1212.35944528 633.6279997825623



if False:

    forest = RandomForestRegressor(n_estimators = 100,criterion='mse',n_jobs=-1,random_state=prng)

    forest = forest.fit(X,fy)

    feature_impportances =  sorted(zip(forest.feature_importances_,train.columns.values),reverse=True)

    #Print the top features

    print(feature_impportances)

    for a,b in feature_impportances:

        print(a,b)

features = ['cat80','cont14','cat101','cont7','cont2','cat79','cat103','cat100','cat12','cat111',

            'cat112','cont8','cont5','cont3','cat81','cont4','cont6','cat53','cont1','cat110',

            'cont13','cont12','cont10','cont11','cat57','cat1','cont9','cat114','cat113']
### XGBoost
import xgboost as xgb



def evalerror(preds, dtrain):

    labels = dtrain.get_label()

    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))



RANDOM_STATE = 1234



params = {

        'min_child_weight': 1,

        'eta': 0.01,

        'colsample_bytree': 0.5,

        'max_depth': 12,

        'subsample': 0.8,

        'alpha': 1,

        'gamma': 1,

        'silent': 1,

        'verbose_eval': True,

        'seed': RANDOM_STATE

    }



xgtrain = xgb.DMatrix(X, label=y)

xgtest = xgb.DMatrix(X_test)



run_cv=False

run_model=False



if run_cv:

    cv = xgb.cv(params, xgtrain, num_boost_round=10, nfold=5, stratified=False,

         early_stopping_rounds=50, verbose_eval=1, show_stdv=True, feval=evalerror, maximize=False)



if run_model:

    model = xgb.train(params, xgtrain, int(2012 / 0.9), feval=evalerror)

    prediction = np.exp(model.predict(xgtest)) - shift