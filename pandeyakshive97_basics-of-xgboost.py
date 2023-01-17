import pandas as pd

import numpy as np

import xgboost as xgb

import sklearn
df = pd.read_csv('../input/mushrooms.csv')

df.head()
df = pd.get_dummies(df)

df.head()
train = np.array(df.loc[:, 'cap-shape_b':])

x_train = train[:int(train.shape[0]*0.8), :]

x_val = train[int(train.shape[0]*0.8):, :]

labels = np.array(df.loc[:, 'class_e':'class_p'])

y_train = labels[:int(train.shape[0]*0.8), :]

y_val = labels[int(train.shape[0]*0.8):, :]

dtrain = xgb.DMatrix(x_train, label = y_train)

dval = xgb.DMatrix(x_val, label = y_val)
params = {

    'objective':'binary:logistic',

    'max_depth':2,

    'silent':1,

    'eta':1

}



num_rounds = 5



watchlist  = [(dval,'val'), (dtrain,'train')]

bst = xgb.train(params, dtrain, num_rounds, watchlist)
tree_dump = bst.get_dump(with_stats = True)
for tree in tree_dump:

    print(tree)
xgb.plot_importance(bst, importance_type='gain', xlabel='Gain')