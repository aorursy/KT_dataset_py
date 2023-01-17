import pandas as pd

import category_encoders as ce

from sklearn import linear_model

from sklearn.model_selection import StratifiedKFold
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



train.sort_index(inplace=True)

train_y = train['target']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)
cat_feat_to_encode = train.columns.tolist()

smoothing=50.0



oof = pd.DataFrame([])

for tr_idx, oof_idx in StratifiedKFold(

    n_splits=5, random_state=1, shuffle=True).split(

        train, train_y):

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])

    oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)



ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

ce_target_encoder.fit(train, train_y)

train = oof.sort_index() 

test = ce_target_encoder.transform(test)
glm = linear_model.LogisticRegression(

  random_state=1, solver='lbfgs', max_iter=5000, fit_intercept=True, 

  penalty='none', verbose=0)



glm.fit(train, train_y)
from datetime import datetime

pd.DataFrame({'id': test_id, 'target': glm.predict_proba(test)[:,1]}).to_csv(

    'sub_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv', 

    index=False)