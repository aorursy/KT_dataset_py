# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn import model_selection

import xgboost as xgb

import datetime



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

yTrain = train['target']

train.drop(['target','id'], axis=1, inplace=True)

id = test['id']

test.drop(['id'],axis=1,inplace=True)





xtrain = train.values

xtest = test.values       

y = yTrain.values
#xgb_test = pd.DataFrame(test[['name']], columns=['name'])

y_pred = np.zeros(xtest.shape[0])

xgtest = xgb.DMatrix(xtest)

score = 0

folds = 10

kf = model_selection.StratifiedKFold(n_splits=folds, shuffle=False, random_state=4)
print('Training and making predictions')

for trn_index, val_index in kf.split(xtrain, y):

    

    xgtrain = xgb.DMatrix(xtrain[trn_index], label=y[trn_index])

    xgvalid = xgb.DMatrix(xtrain[val_index], label=y[val_index])

    

    params = {

        'eta': 0.05, #0.03

        'silent': 1,

        'verbose_eval': True,

        'verbose': False,

        'seed': 4

    }

    params['objective'] = 'binary:logistic'

    params['eval_metric'] = "auc"

    params['min_child_weight'] = 15

    params['cosample_bytree'] = 0.8

    params['cosample_bylevel'] = 0.9

    params['max_depth'] = 4

    params['subsample'] = 0.9

    params['max_delta_step'] = 10

    params['gamma'] = 1

    params['alpha'] = 0

    params['lambda'] = 1

    #params['base_score'] =  0.63

    

    watchlist = [ (xgtrain,'train'), (xgvalid, 'valid') ]

    model = xgb.train(list(params.items()), xgtrain, 5000, watchlist, 

                      early_stopping_rounds=25, verbose_eval = 50)

    

    y_pred += model.predict(xgtest,ntree_limit=model.best_ntree_limit)

    score += model.best_score



y_pred /= folds

score /= folds

print('Mean AUC:',score)



now = datetime.datetime.now()

my_submission = pd.DataFrame({'id':id , 'target': y_pred})
my_submission.to_csv('submission.csv', index=False)