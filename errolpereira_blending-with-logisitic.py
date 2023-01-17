!pip install the1owl
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from ml import simple #the1owl

import gc



import warnings

warnings.filterwarnings('ignore')



PATH = '/kaggle/input/cat-in-the-dat/'
#Reading the dataset.

train = pd.read_csv(f'{PATH}train.csv')

test = pd.read_csv(f'{PATH}test.csv')
%%time

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id', 'bin_0'], axis=1, inplace=True)

test.drop(['id','bin_0'], axis=1, inplace=True)



print(train.shape)

print(test.shape)
%%time

traintest = pd.concat([train, test])

dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)

train_ohe = dummies.iloc[:train.shape[0], :]

test_ohe = dummies.iloc[train.shape[0]:, :]



print(train_ohe.shape)

print(test_ohe.shape)
%%time

train_ohe = train_ohe.sparse.to_coo().tocsr()

test_ohe = test_ohe.sparse.to_coo().tocsr()
train_ohe.shape, test_ohe.shape
print('#'*20)

print('StratifiedKFold training...')



# Same as normal kfold but we can be sure

# that our target is perfectly distribuited

# over folds





folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)



#initializing the model

model = LogisticRegression(solver='lbfgs', max_iter=200, C=0.095)



#score.

score = []



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_ohe, target, groups=target)):

    print('Fold:',fold_+1)

    tr_x, tr_y = train_ohe[trn_idx,:], target[trn_idx]    

    vl_x, v_y = train_ohe[val_idx,:], target[val_idx]

    

    #fitting on training data.

    %time model.fit(tr_x, tr_y)

    

    #predicting on test.

    y_pred = model.predict(vl_x)

    

    #storing score

    score.append(roc_auc_score(v_y, y_pred))

    print(f'AUC score : {roc_auc_score(v_y, y_pred)}')



print('Average AUC score', np.mean(score))

print('#'*20)
#fitting on the entire data.

%time model.fit(train_ohe, target)
#making predictions on test data.

pred_test = model.predict_proba(test_ohe)[:,0]
#submission file.

sub = pd.read_csv(f'{PATH}sample_submission.csv')

# #reseting index

# test_df = test_df.reset_index()

sub['target'] = pred_test

sub.to_csv('log_reg_0.1.csv', index=None, header=True)
#https://www.kaggle.com/peterhurford/why-not-logistic-regression <-- Upvote here

sub = simple.Blend(['../input/why-not-logistic-regression/submission.csv', 'log_reg_0.1.csv'], 'id', 'target', 'blend01.csv')
#https://www.kaggle.com/abhishek/entity-embeddings-to-handle-categories <-- Upvote here

sub = simple.Blend(['../input/entity-embeddings-to-handle-categories/submission.csv', 'blend01.csv'], 'id', 'target', 'blend02.csv')
#https://www.kaggle.com/ccccat/r-glmnet <-- Upvote here

sub = simple.Blend(['../input/r-glmnet/submission_glmnet.csv', 'blend02.csv'], 'id', 'target', 'blend03.csv')