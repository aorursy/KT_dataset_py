# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print('Load data...')
train = pd.read_csv("../input/train.csv")
target = train['target']
train = train.drop(['ID','target'],axis=1)
test = pd.read_csv("../input/test.csv")
ids = test['ID'].values
test = test.drop(['ID'],axis=1)
print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            train.loc[train_series.isnull(), train_name] = 0 #train_series.mean()
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = 0 #train_series.mean()  #TODO
xgtrain = xgb.DMatrix(train.values, target.values)
xgtest = xgb.DMatrix(test.values)
xgboost_params = { 
   "objective": "binary:logistic",
   "booster": "gbtree",
   "eta": 0.017483,
   "min_child_weight": 4.436,
   "subsample": 0.812,
   "colsample_bytree": 0.844,
   "max_depth": 5,
   "gamma":0.00036354432647887241
}

clf = xgb.train(xgboost_params
                , xgtrain
                , num_boost_round=500
                , verbose_eval=True
                , maximize=False)

train_preds = clf.predict(xgtrain, ntree_limit=clf.best_iteration)
a = np.transpose(np.vstack([target, train_preds]))
a

predictions_file = open("result.csv", "w")
open_file_object = csv.writer(predictions_file)
#open_file_object.writerow(["ID", "PredictedProb"])
#open_file_object.writerows(zip(ids, test_preds))
open_file_object.writerows(a)
predictions_file.close()