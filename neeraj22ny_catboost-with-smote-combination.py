# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df =  pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
r1 = train_df.shape[0] ; r2  = test_df.shape[0]

c1 = train_df.shape[1]; c2 = test_df.shape[1]

print("Train Data has {0} number of rows and {1} of columns".format(r1,c1))

print("Test Data has {0} number of rows and {1} of columns".format(r2,c2))
train_df.head()
train_df.target.value_counts()
import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
ax = sns.countplot(x="target", data=train_df, palette = "Set1")
X = train_df.drop(['ID_code','target'],axis =1)

y = train_df['target']

test = test_df.drop(['ID_code'],axis =1)
#standardizing Variables 

#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()

#scaler.fit(X)

#X_scaled = scaler.transform(X)

#X_test_scaled = scaler.transform(test)

#X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

#X_test_scaled = pd.DataFrame(X_test_scaled, columns=test.columns, index=test.index)
X.head()
y.head()
test.head()
print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))
#Perform Upsampling Using SMOTE

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2)

X_res, y_res = sm.fit_sample(X, y.ravel())



print('After OverSampling, the shape of X: {}'.format(X_res.shape))

print('After OverSampling, the shape of y: {} \n'.format(y_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_res==0)))
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from catboost import Pool, CatBoostClassifier
model = CatBoostClassifier(loss_function="Logloss",

                           eval_metric="AUC",

                           learning_rate=0.01,

                           iterations=10000,

                           random_seed=42,

                           od_type="Iter",

                           depth=10,

                           task_type = 'GPU',

                           early_stopping_rounds=100

                          )
n_split = 5

kf = KFold(n_splits=n_split, random_state=42, shuffle=True)
y_valid_pred = 0*y_res

y_test_pred = 0
y_valid_pred.shape
for idx, (train_index, valid_index) in enumerate(kf.split(X_res)):

    y_train, y_valid = y_res[train_index], y_res[valid_index]

    X_train, X_valid = X_res[train_index,:], X_res[valid_index,:]

    _train = Pool(X_train, label=y_train)

    _valid = Pool(X_valid, label=y_valid)

    print( "\nFold ", idx)

    fit_model = model.fit(_train,

                          eval_set=_valid,

                          use_best_model=True,

                          verbose=200,

                          plot=True

                         )

    pred = fit_model.predict_proba(X_valid)[:,1]

    print( "  auc = ", roc_auc_score(y_valid, pred) )

    y_valid_pred[valid_index] = pred

    y_test_pred += fit_model.predict_proba(test)[:,1]

y_test_pred /= n_split
from pathlib import Path

root = Path("../input")
submission = pd.read_csv(root.joinpath("sample_submission.csv"))

submission['target'] = y_test_pred

submission.to_csv('submission3.csv', index=False)

submission[:10]



#submission = pd.DataFrame(y_test_pred)

#submission[:10]