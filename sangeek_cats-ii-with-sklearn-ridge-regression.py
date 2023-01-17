from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import Ridge



import pandas as pd

import numpy as np





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/multi-cat-encodings/X_train_te.csv')

test = pd.read_csv('../input/multi-cat-encodings/X_test_te.csv')

sample_submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')
y = train['target'].values

train = train.drop(['target','fold_column'], axis=1)

X = train.values.copy()

test_np = test.values.copy()
train_oof = np.zeros((train.shape[0],))

test_preds = 0

train_oof.shape
%%time

n_splits = 5

kf = KFold(n_splits=n_splits, random_state=17, shuffle=True)

scores = []



for jj, (train_index, val_index) in enumerate(kf.split(X)):

    print("Fitting fold", jj+1)

    X_train, X_test = X[train_index], X[val_index]

    y_train, y_test = y[train_index], y[val_index]



    #model = LinearRegression(use_gpu=True, regularizer = 1.0/5, dual=False)

    model = Ridge(alpha = 5)

    #model = LinearRegression(use_gpu=True, dual=False)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    train_oof[val_index] = y_pred

    score = roc_auc_score(y_test, y_pred)

    print("Fold AUC:", score)

    scores.append(score)

    #test_preds += model.predict(test).values/n_splits

    test_preds += model.predict(test_np)/n_splits

    

print("Mean AUC:", np.mean(scores))
sample_submission['target'] = test_preds

sample_submission.to_csv('submission.csv', index=False)
np.save('test_preds', test_preds)

np.save('train_oof', train_oof)