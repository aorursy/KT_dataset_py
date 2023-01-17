# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")
train.head(10)
constant_columns = []
for col in train.columns:
    if(train[col].std() == 0):
        constant_columns.append(col)
        
print("%d columns are constant" % (len(constant_columns)))
print(constant_columns)

train.drop(constant_columns , axis = 1 , inplace = True)
classes = train['Cover_Type'].unique()
print("Number of classes : %d" % (len(classes)))
print(classes)
train['Cover_Type'] = train['Cover_Type'].apply(lambda x : int(x - 1))
features_columns = [col for col in train.columns if col not in ['Id' , 'Cover_Type']]
target = 'Cover_Type'
num_classes = len(classes)
X , Y = train[features_columns] , train[target]
X_train , X_val , y_train , y_val = train_test_split(X , Y , test_size = 0.2 , random_state = 0)
param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.1
param['max_depth'] = 7
param['eval_metric'] = 'mlogloss'
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 0
param['alpha'] = 0.001
param['num_class'] = num_classes

params = list(param.items())

xgtrain = xgb.DMatrix(X_train , label = y_train)
xgval = xgb.DMatrix(X_val , label = y_val)
xgtest = xgb.DMatrix(test[features_columns])
watch_list = [(xgtrain , 'train') , (xgval , 'val')]

model = xgb.train(params, xgtrain, 500, watch_list, early_stopping_rounds=20)
val_pred = model.predict(xgval , ntree_limit = model.best_ntree_limit)
val_acc = accuracy_score(val_pred , y_val)
print("Validation accuracy {}%".format(val_acc))
y_test = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
submission['Cover_Type'] = y_test
submission['Cover_Type'] = submission['Cover_Type'].apply(lambda x : int(x + 1))
submission.to_csv("submission.csv" , index = False)

