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
%%time



# Load data

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



print(train.shape)

print(test.shape)
%%time



# Subset

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)
%%time



# One Hot Encode

from sklearn.preprocessing import OneHotEncoder

traintest = pd.concat([train, test])

one=OneHotEncoder()

one.fit(traintest)

train_data = traintest.iloc[:train.shape[0], :]

test_data = traintest.iloc[train.shape[0]:, :]

train_ohe=one.transform(train_data)

test_ohe=one.transform(test_data)



print(train_ohe.shape)

print(test_ohe.shape)
%%time



from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression



# Model

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = KFold(n_splits=5)

    fold_splits = kf.split(train, target)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/5')

        dev_X, val_X = train[dev_index], train[val_index]

        dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()

        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index] = pred_val_y

        if eval_fn is not None:

            cv_score = eval_fn(val_y, pred_val_y)

            cv_scores.append(cv_score)

            print(label + ' cv score {}: {}'.format(i, cv_score))

        i += 1

    print('{} cv scores : {}'.format(label, cv_scores))

    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv std score : {}'.format(label, np.std(cv_scores)))

    pred_full_test = pred_full_test / 5.0

    results = {'label': label,

              'train': pred_train, 'test': pred_full_test,

              'cv': cv_scores}

    return results





def runLR(train_X, train_y, test_X, test_y, test_X2, params):

    print('Train LR')

    model = LogisticRegression(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict_proba(test_X)[:, 1]

    print('Predict 2/2')

    pred_test_y2 = model.predict_proba(test_X2)[:, 1]

    return pred_test_y, pred_test_y2





lr_params = {'solver': 'lbfgs', 'C': 0.1}

results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, roc_auc_score, 'lr')
%%time



# We now have a model with a CV score of 0.8032. Nice! Let's submit that



# Make submission

submission = pd.DataFrame({'id': test_id, 'target': results['test']})

submission.to_csv('submission.csv', index=False)