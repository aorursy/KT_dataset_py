import numpy as np

import pandas as pd

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

import xgboost as xgb
train_ds = pd.read_csv('../input/train.csv')

test_ds = pd.read_csv('../input/test.csv')

print(train_ds.shape, test_ds.shape)
train_y = train_ds['target']

train_x = train_ds.drop(['target', 'ID_code'], axis=1)

id_test = test_ds['ID_code']

test_x = test_ds.drop(['ID_code'], axis=1)



train_x = scale(train_x)

test_x = scale(test_x)



train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.25, random_state=82)

print(train_x.shape, train_y.shape)
param = {

    'objective': 'multi:softmax',

    'num_class': 2,

    'tree_method': 'gpu_hist'

}



dtrain = xgb.DMatrix(train_x, label=train_y)

ddev = xgb.DMatrix(dev_x, label=dev_y)

dtest = xgb.DMatrix(test_x)

bst = xgb.train(param, dtrain, 500, evals=[(ddev, 'dev')])
prediction = bst.predict(dtest)
prediction = np.where(prediction > 0.5, 1, 0)
pd.DataFrame({"ID_code":id_test,"target":prediction}).to_csv('result_keras.csv',index=False,header=True)