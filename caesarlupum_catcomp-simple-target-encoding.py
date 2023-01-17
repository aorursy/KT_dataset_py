import pandas as pd

import numpy as np

import category_encoders as ce

import lightgbm as lgb

from sklearn import linear_model

from sklearn.model_selection import StratifiedKFold

import gc
# Suppress warnings 

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import HTML

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')

print(train.target.value_counts()[0]/300000, train.target.value_counts()[1]/300000, )

train.sort_index(inplace=True)

train_y = train['target']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)
train
HTML('<iframe width="680" height="620" src="https://www.youtube.com/embed/8odLEbSGXoI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')

from sklearn.metrics import roc_auc_score

cat_feat_to_encode = train.columns.tolist()

# target =0  0.69412%, target =1 0.30588

smoothing=0.50



oof = pd.DataFrame([])

for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=1, shuffle=True).split(train, train_y):

    

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])

    oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)



    

    

    

ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

ce_target_encoder.fit(train, train_y)

train = oof.sort_index() 

test = ce_target_encoder.transform(test)
glm = linear_model.LogisticRegression(

  random_state=1, solver='lbfgs', max_iter=2019, fit_intercept=True, 

  penalty='none', verbose=0)



glm.fit(train, train_y)
from datetime import datetime

pd.DataFrame({'id': test_id, 'target': glm.predict_proba(test)[:,1]}).to_csv(

    'sub_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv', 

    index=False)