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



import xgboost as xgb

from itertools import product

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error as mse
tr = pd.read_csv('../input/train.csv')

te = pd.read_csv('../input/test.csv')



columns = ['LotArea','LotShape','LandContour','LandSlope','LotConfig','LotFrontage']

features = columns[:-1]

target = [columns[-1]]



df = tr.loc[tr['LotFrontage'].notnull(),columns].append(\

        te.loc[te['LotFrontage'].notnull(),columns])



for feature in ['LotShape','LandContour','LandSlope','LotConfig']:

    le = LabelEncoder()

    le.fit(df[feature])

    df[feature] = le.transform(df[feature])

 
assert(df.isnull().sum().sum()==0)
np.sqrt(np.mean((df['LotFrontage']-df['LotFrontage'].median())**2))
eta_list = [0.1,0.2,0.3] 

max_depth_list = [4,6,8,10]

subsample = 0.8

colsample_bytree = 0.4



num_boost_round = 400 

early_stopping_rounds = 20

test_size = 0.1665 # roughly corresponds the fraction of missing lotfrontage



opt = 1e8

optparam = None

repeat = 5
for eta,max_depth in product(eta_list, max_depth_list): 

    params = {

        "objective": "reg:linear",

        "booster" : "gbtree", 

        "eval_metric": "rmse", 

        "eta": eta, 

        "tree_method": 'exact',

        "max_depth": max_depth,

        "subsample": subsample, 

        "colsample_bytree": colsample_bytree,

        "silent": 1,

        "seed": 0,

    }



    scores, rounds = [], []

    for _ in range(repeat):

        X_train, X_valid = train_test_split(df, test_size=test_size)

        dtrain = xgb.DMatrix(X_train[features].as_matrix(), X_train[target].as_matrix()) 

        dvalid = xgb.DMatrix(X_valid[features].as_matrix(), X_valid[target].as_matrix())

    

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')] 

        model = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \

                early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

        scores.append(model.best_score)

        rounds.append(model.best_iteration)



    print('eta {0} depth {1} scores {2} rounds {3}'.format(eta,max_depth,scores,rounds))

    if np.mean(scores)<opt:

        opt = np.mean(scores)

        optparam = (eta,max_depth,int(np.mean(rounds))) # taking average for rounds isn't exactly well-motivated mathematically

 

print(opt,optparam)

params = {

    "objective": "reg:linear",

    "booster" : "gbtree", 

    "eval_metric": "rmse", 

    "eta": optparam[0], 

    "tree_method": 'exact',

    "max_depth": optparam[1],

    "subsample": subsample, 

    "colsample_bytree": colsample_bytree,

    "silent": 1,

    "seed": 0,

}



X_train, X_valid = train_test_split(df, test_size=test_size)

dtrain = xgb.DMatrix(X_train[features].as_matrix(), X_train[target].as_matrix()) 

dvalid = xgb.DMatrix(X_valid[features].as_matrix(), X_valid[target].as_matrix())

watchlist = [(dtrain, 'train')] 

model_count = 5

models = []

for i in range(model_count):

    params["seed"] = i

    models.append(xgb.train(params, dtrain, optparam[2], evals=watchlist, verbose_eval=False))



for i,model in enumerate(models):

    X_train['LotFrontagePred{0}'.format(i)] = model.predict(dtrain)

    X_valid['LotFrontagePred{0}'.format(i)] = model.predict(dvalid)



X_train['LotFrontagePred'] = sum([X_train['LotFrontagePred{0}'.format(i)] for i in range(model_count)]) / model_count

X_valid['LotFrontagePred'] = sum([X_valid['LotFrontagePred{0}'.format(i)] for i in range(model_count)]) / model_count

print('Train RMSE {0:.6f}, Valid RMSE {1:.6f}'.format(\

        np.sqrt(mse(X_train['LotFrontage'], X_train['LotFrontagePred'])), \

        np.sqrt(mse(X_valid['LotFrontage'], X_valid['LotFrontagePred']))))
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(1)

f, axarr = plt.subplots(2,1,figsize=(10,18))

axarr[0].scatter(X_train['LotFrontage'],X_train['LotFrontagePred'])

axarr[1].scatter(X_valid['LotFrontage'],X_valid['LotFrontagePred'])
fscore = []

for fn in models[0].get_fscore():

    fs = sum(model.get_fscore()[fn] for model in models)

    fscore.append((features[int(fn[1:])],fs))



fscore = sorted(fscore, key = lambda x:-x[1])

for u in fscore:

    print(u[0],u[1])


