# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV, cross_validate, KFold



import xgboost as xgb

import matplotlib.pyplot as plt

train = pd.read_csv("../input/santander-customer-satisfaction/train.csv")

test = pd.read_csv("../input/santander-customer-satisfaction/test.csv")
remove = []

for col in train.columns:

    if train[col].std() == 0:

        remove.append(col)

        

train.drop(remove, axis=1, inplace=True)

test.drop(remove, axis=1, inplace=True)
remove = []

cols = train.columns

for i in range(len(cols)-1):

    v = train[cols[i]].values

    for j in range(i+1,len(cols)):

        if np.array_equal(v,train[cols[j]].values):

            remove.append(cols[j])

            

train.drop(remove, axis=1, inplace=True)

test.drop(remove, axis=1, inplace=True)

train = train.replace(-999999,2)

test = test.replace(-999999,2)
test_id = test.ID

test = test.drop(["ID"],axis=1)
train_df_0 = train[train['TARGET'] == 0]

train_df_1 = train[train['TARGET'] == 1]
train_dfs = []

target_df_length = len(train_df_1)

for i in range(len(train_df_0)//(target_df_length*2)):

    item_df = train_df_0[target_df_length*2*i:target_df_length*2*(i+1)]

    train_dfs.append(item_df)
from sklearn.utils import shuffle

# for i in train_dfs[:1]:

#     train_data_x = pd.concat([i,train_df_1]).drop(['TARGET','ID'],axis=1)

#     train_data_y = [0]*len(i) + [1]*len(train_df_1)

#     X, y = shuffle(train_data_x.values, train_data_y, random_state=0)

#     model = xgb.XGBClassifier(max_depth = 5, n_estimators=160, learning_rate=0.02,nthread=4,

#                 subsample=0.95, colsample_bytree=0.85)

#     gscv = cross_validate(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)['test_score']

#     print('Use model:{}'.format(model.__class__.__name__))

#     print('Mean AUC:{:.5f}'.format(gscv.mean()))
models = []

preds = []

for i in train_dfs:

    train_data_x = pd.concat([i,train_df_1]).drop(['TARGET','ID'],axis=1)

    train_data_y = [0]*len(i) + [1]*len(train_df_1)

    X, y = shuffle(train_data_x.values, train_data_y, random_state=0)

    model = xgb.XGBClassifier(max_depth = 5, n_estimators=160, learning_rate=0.02,nthread=4,

            subsample=0.95, colsample_bytree=0.85) #0.840 0.836 0.840 0.840 0.838

    model.fit(X, y)

    models.append(model)

    pred = model.predict_proba(test.values)

    preds.append(pred[:,1])
res = pd.DataFrame(preds).T

res['sum'] = res.sum(axis=1)

res['res1'] = res['sum']/12

submission = pd.DataFrame({"ID":test_id, "TARGET": res['res1']})

submission.to_csv("submission.csv", index=False)