%%time



import pandas as pd

import numpy as np

import category_encoders as ce



# Load data

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



print(train.shape)

print(test.shape)
train.head()
from sklearn.model_selection import StratifiedKFold
# from pandas.api.types import CategoricalDtype 



# ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

#                                      'Master', 'Grandmaster'], ordered=True)

# ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

#                                      'Boiling Hot', 'Lava Hot'], ordered=True)

# ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

#                                      'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

# ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

#                                      'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

#                                      'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)



# train[["ord_1", "ord_2", "ord_3", "ord_4"]] = train[["ord_1", "ord_2", "ord_3", "ord_4"]].astype("category")

# test[["ord_1", "ord_2", "ord_3", "ord_4"]] = test[["ord_1", "ord_2", "ord_3", "ord_4"]].astype("category")

# train["ord_1"] = train.ord_1.astype(ord_1)

# train["ord_2"] = train.ord_2.astype(ord_2)

# train["ord_3"] = train.ord_3.astype(ord_3)

# train["ord_4"] = train.ord_4.astype(ord_4)



# test["ord_1"] = test.ord_1.cat.codes

# test["ord_2"] = test.ord_2.cat.codes

# test["ord_3"] = test.ord_3.cat.codes

# test["ord_4"] = test.ord_4.cat.codes
%%time



# Subset

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)
test.head()
%%time



# One Hot Encode

traintest = pd.concat([train, test])

dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)

train_ohe = dummies.iloc[:train.shape[0], :]

test_ohe = dummies.iloc[train.shape[0]:, :]



print(train_ohe.shape)

print(test_ohe.shape)
train_ohe.head(2)
cat_feat_to_encode = train.columns.tolist()

smoothing=50.0



oof = pd.DataFrame([])

for tr_idx, oof_idx in StratifiedKFold(

    n_splits=5, random_state=1, shuffle=True).split(

        train, target):

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(train.iloc[tr_idx, :], target.iloc[tr_idx])

    oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)



ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

ce_target_encoder.fit(train, target)

train_enc = oof.sort_index() 

test_enc = ce_target_encoder.transform(test)
train_enc.shape
%%time



# To be honest, I am a bit confused what is going on with the new sparse dataframe interface in Pandas v0.25



# It looks like `sparse = True` in `get_dummies` no longer makes anything sparse, and we have to explicitly convert

# like this...



# If you don't do this, the model takes forever... it is much much faster on sparse data!



train_ohe = train_ohe.sparse.to_coo().tocsr()

test_ohe = test_ohe.sparse.to_coo().tocsr()
import scipy

scipy.sparse.hstack
train_merge = scipy.sparse.hstack([train_ohe, train_enc.values]).tocsr()

test_merge = scipy.sparse.hstack([test_ohe, test_enc.values]).tocsr()
%%time



from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression



# Model

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = KFold(n_splits=10)

    fold_splits = kf.split(train, target)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/10')

        #print("dev_index", dev_index)

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



from catboost import CatBoostClassifier

import lightgbm as lgb

def runCat(train_X, train_y, test_X, test_y, test_X2, params):

    print('Train LR')

    model = CatBoostClassifier(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict_proba(test_X)[:, 1]

    print('Predict 2/2')

    pred_test_y2 = model.predict_proba(test_X2)[:, 1]

    return pred_test_y, pred_test_y2



def runLgb(train_X, train_y, test_X, test_y, test_X2, params):

    model = lgb.train(

    params={

        'max_depth': 3, 

        'num_leaves': 150,

        'reg_alpha': 0.6, 

        'reg_lambda': 0.6,

        'objective': 'binary',

        "boosting_type": "gbdt",

        "metric": 'auc',

        "verbosity": -1,

        'random_state': 1,

        'lr': 0.01

    },

    train_set=lgb.Dataset(train_X, label=train_y),

    num_boost_round=700)

    

    print('Predict 1/2')

    pred_test_y = model.predict(test_X)

    print('Predict 2/2')

    pred_test_y2 = model.predict(test_X2)

    return pred_test_y, pred_test_y2





model = CatBoostClassifier(learning_rate=0.006, iterations=1000, thread_count=32,

                           eval_metric='Accuracy')



lr_params = {'solver': 'lbfgs', 'C': 0.1, "max_iter":500, 'thread_count':32, "eval_metric":"AUC"}

cat_params = {"iterations":500, 'learning_rate': 0.006, 'thread_count':32, 'eval_metric':'AUC'}

#results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr')

#results = run_cv_model(train_merge, test_merge, target, runLR, lr_params, auc, 'lr')

#results = run_cv_model(train_merge.toarray(), test_merge.toarray(), target, runCat, cat_params, auc, 'catboost')

results = run_cv_model(train_merge, test_merge, target, runLgb, {}, auc, 'lgb')
# Make submission

submission = pd.DataFrame({'id': test_id, 'target': results['test']})

submission.to_csv('submission.csv', index=False)