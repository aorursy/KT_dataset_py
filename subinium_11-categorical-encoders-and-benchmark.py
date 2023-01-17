# If you want to test this on your local notebook

# http://contrib.scikit-learn.org/categorical-encoding/

# !pip install category-encoders
import pandas as pd



from category_encoders.ordinal import OrdinalEncoder

from category_encoders.woe import WOEEncoder

from category_encoders.target_encoder import TargetEncoder

from category_encoders.sum_coding import SumEncoder

from category_encoders.m_estimate import MEstimateEncoder

from category_encoders.leave_one_out import LeaveOneOutEncoder

from category_encoders.helmert import HelmertEncoder

from category_encoders.cat_boost import CatBoostEncoder

from category_encoders.james_stein import JamesSteinEncoder

from category_encoders.one_hot import OneHotEncoder



TEST = False
%%time

train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)
feature_list = list(train.columns) # you can custumize later.
%%time

LE_encoder = OrdinalEncoder(feature_list)

train_le = LE_encoder.fit_transform(train)

test_le = LE_encoder.transform(test)
# %%time

# this method didn't work because of RAM memory. 

# so we have to use pd.dummies 

# OHE_encoder = OneHotEncoder(feature_list)

# train_ohe = OHE_encoder.fit_transform(train)

# test_ohe = OHE_encoder.transform(test)
# %%time

# this method didn't work because of RAM memory. 

# SE_encoder =SumEncoder(feature_list)

# train_se = SE_encoder.fit_transform(train[feature_list], target)

# test_se = SE_encoder.transform(test[feature_list])
# %%time

# this method didn't work because of RAM memory. 

# HE_encoder = HelmertEncoder(feature_list)

# train_he = HE_encoder.fit_transform(train[feature_list], target)

# test_he = HE_encoder.transform(test[feature_list])
%%time



TE_encoder = TargetEncoder()

train_te = TE_encoder.fit_transform(train[feature_list], target)

test_te = TE_encoder.transform(test[feature_list])



train_te.head()
%%time

MEE_encoder = MEstimateEncoder()

train_mee = MEE_encoder.fit_transform(train[feature_list], target)

test_mee = MEE_encoder.transform(test[feature_list])
%%time

WOE_encoder = WOEEncoder()

train_woe = WOE_encoder.fit_transform(train[feature_list], target)

test_woe = WOE_encoder.transform(test[feature_list])
%%time

JSE_encoder = JamesSteinEncoder()

train_jse = JSE_encoder.fit_transform(train[feature_list], target)

test_jse = JSE_encoder.transform(test[feature_list])
%%time

LOOE_encoder = LeaveOneOutEncoder()

train_looe = LOOE_encoder.fit_transform(train[feature_list], target)

test_looe = LOOE_encoder.transform(test[feature_list])
%%time

CBE_encoder = CatBoostEncoder()

train_cbe = CBE_encoder.fit_transform(train[feature_list], target)

test_cbe = CBE_encoder.transform(test[feature_list])
%%time

import gc

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression



encoder_list = [ OrdinalEncoder(), WOEEncoder(), TargetEncoder(), MEstimateEncoder(), JamesSteinEncoder(), LeaveOneOutEncoder() ,CatBoostEncoder()]



X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=97)



for encoder in encoder_list:

    print("Test {} : ".format(str(encoder).split('(')[0]), end=" ")

    train_enc = encoder.fit_transform(X_train[feature_list], y_train)

    #test_enc = encoder.transform(test[feature_list])

    val_enc = encoder.transform(X_val[feature_list])

    lr = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)

    lr.fit(train_enc, y_train)

    lr_pred = lr.predict_proba(val_enc)[:, 1]

    score = auc(y_val, lr_pred)

    print("score: ", score)

    del train_enc

    del val_enc

    gc.collect()



%%time

from sklearn.model_selection import KFold

import numpy as np



# CV function original : @Peter Hurford : Why Not Logistic Regression? https://www.kaggle.com/peterhurford/why-not-logistic-regression



def run_cv_model(train, test, target, model_fn, params={}, label='model'):

    kf = KFold(n_splits=5)

    fold_splits = kf.split(train, target)



    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for dev_index, val_index in fold_splits:

        print('Started {} fold {}/5'.format(label, i))

        dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]

        dev_y, val_y = target[dev_index], target[val_index]

        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index] = pred_val_y

        cv_score = auc(val_y, pred_val_y)

        cv_scores.append(cv_score)

        print(label + ' cv score {}: {}'.format(i, cv_score))

        i += 1

        

    print('{} cv scores : {}'.format(label, cv_scores))

    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv std score : {}'.format(label, np.std(cv_scores)))

    pred_full_test = pred_full_test / 5.0

    results = {'label': label, 'train': pred_train, 'test': pred_full_test, 'cv': cv_scores}

    return results





def runLR(train_X, train_y, test_X, test_y, test_X2, params):

    model = LogisticRegression(**params)

    model.fit(train_X, train_y)

    pred_test_y = model.predict_proba(test_X)[:, 1]

    pred_test_y2 = model.predict_proba(test_X2)[:, 1]

    return pred_test_y, pred_test_y2

if TEST:



    lr_params = {'solver': 'lbfgs', 'C': 0.1}



    results = list()



    for encoder in  [ OrdinalEncoder(), WOEEncoder(), TargetEncoder(), MEstimateEncoder(), JamesSteinEncoder(), LeaveOneOutEncoder() ,CatBoostEncoder()]:

        train_enc = encoder.fit_transform(train[feature_list], target)

        test_enc = encoder.transform(test[feature_list])

        result = run_cv_model(train_enc, test_enc, target, runLR, lr_params, str(encoder).split('(')[0])

        results.append(result)

    results = pd.DataFrame(results)

    results['cv_mean'] = results['cv'].apply(lambda l : np.mean(l))

    results['cv_std'] = results['cv'].apply(lambda l : np.std(l))

    results[['label','cv_mean','cv_std']].head(8)
if TEST:

    for idx, label in enumerate(results['label']):

        sub_df = pd.DataFrame({'id': test_id, 'target' : results.iloc[idx]['test']})

        sub_df.to_csv("LR_{}.csv".format(label), index=False)


