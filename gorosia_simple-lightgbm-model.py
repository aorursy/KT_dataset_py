import pandas as pd

import numpy as np

import lightgbm as lgb

from lightgbm import LGBMModel,LGBMClassifier

from sklearn import metrics
Train = pd.read_csv("../input/train.csv",na_values=-1)

Test = pd.read_csv("../input/test.csv",na_values=-1)
## Filling the missing data NAN with 0

Train_median = pd.DataFrame()

for column in Train.columns:

    Train_median[column] = Train[column].fillna(0)



Train = Train_median.copy()
## Filling the missing data NAN with 0

Test_median = pd.DataFrame()

for column in Test.columns:

    Test_median[column] = Test[column].fillna(0)



Test = Test_median.copy()
one_hot = {c: list(Train[c].unique()) for c in Train.columns if c not in ['id','target']}
for c in one_hot:

    if len(one_hot[c])>2 and len(one_hot[c]) < 7:

        for val in one_hot[c]:

            Train[c+'_oh_' + str(val)] = (Train[c].values == val).astype(np.int)

            

for c in one_hot:

    if len(one_hot[c])>2 and len(one_hot[c]) < 7:

        for val in one_hot[c]:

            Test[c+'_oh_' + str(val)] = (Test[c].values == val).astype(np.int)
def gini(y, pred):

    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)

    g = 2 * metrics.auc(fpr, tpr) -1

    return g



def gini_lgb(preds, dtrain):

    y = list(dtrain.get_label())

    score = gini(y, preds) / gini(y, y)

    return 'gini', score, True
# Process data

id_test = Test['id'].values

id_train = Train['id'].values



for c in Train.select_dtypes(include=['float64']).columns:

    Train[c]=Train[c].astype(np.float32)

    Test[c]=Test[c].astype(np.float32)

for c in Train.select_dtypes(include=['int64']).columns[2:]:

    Train[c]=Train[c].astype(np.int8)

    Test[c]=Test[c].astype(np.int8)

    

y = Train['target']

X = Train.drop(['target', 'id'], axis=1)

y_valid_pred = 0*y

X_test = Test.drop(['id'], axis=1)

y_test_pred = 0
Train_lgb = lgb.Dataset(X,label=y)
params = {

    'max_depth' : 5,

    'num_leaves' : 31,

    'objective' : 'binary',

    'metric' : 'auc'

}
#cv_result = lgb.cv(params, Train_lgb, num_boost_round = 150, nfold=5, seed=94, feval=gini_lgb)

#cv_result['gini-mean'].index(max(cv_result['gini-mean']))

#n_estimators = 90
Lgb = LGBMClassifier(n_estimators=90, silent=False, random_state =94, max_depth=5,num_leaves=31,objective='binary',metrics ='auc')
fit_model = Lgb.fit(X, y,eval_metric=gini_lgb)
# Create submission file

y_test_pred = fit_model.predict_proba(X_test)[:,1]

#sub = pd.DataFrame()

#sub['id'] = id_test

#sub['target'] = y_test_pred

#sub.to_csv('BaLgb_submit.csv', float_format='%.6f', index=False)

#It LB = 0.275~0.279