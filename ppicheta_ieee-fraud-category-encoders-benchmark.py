import gc

import os

import random

import time



import category_encoders as ce

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.base import clone

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from tqdm import tqdm

from xgboost import XGBClassifier
SEED = 42

random.seed(SEED)

os.environ['PYTHONHASHSEED'] = str(SEED)

np.random.seed(SEED)
input_dir = '../input/fraud-data/fraud_data.h5'
train = pd.read_hdf(input_dir, key='train')

test = pd.read_hdf(input_dir, key='test')
train.shape
test.shape
ntrain = train.shape[0]

ntest = test.shape[0]
X = train.drop('isFraud', axis=1)

y = train['isFraud'].copy()

del train

gc.collect()
train_test = pd.concat((X, test), sort=False)

del X, test

gc.collect()
cat_fea = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',

           'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',

           'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',

           'DeviceType', 'DeviceInfo'] + ['id_' + str(i) for i in range(12, 39)]
gc.collect()
train_test = train_test[cat_fea]

gc.collect()
for col in tqdm(train_test.columns):

    if train_test[col].dtype == 'float64':

        train_test[col] = train_test[col].astype(np.float32)

    if (train_test[col].dtype == 'int64'):

        train_test[col] = train_test[col].astype(np.int32)
gc.collect()
train_test.shape
val_set = train_test.iloc[ntrain - int(0.2 * ntrain):ntrain, :].index



X_train = train_test.iloc[:ntrain, :]

X_test = train_test.iloc[ntrain:, :]



X_train_sub = X_train[~X_train.index.isin(val_set)]

y_train = y[~y.index.isin(val_set)]



X_val = X_train[X_train.index.isin(val_set)]

y_val = y[y.index.isin(val_set)]



del val_set

gc.collect()
def none_validation(encoder, X, y, X_test, epochs):

    cv = KFold(n_splits=epochs, shuffle=True, random_state=SEED)



    loc_encoder = clone(encoder)

    loc_encoder.fit(X, y)

    

    X_test_enc = loc_encoder.transform(X_test)

    X_enc = loc_encoder.transform(X)

    for tr_idx, val_idx in cv.split(X_enc, y.values): 

        X_tr, X_vl = X_enc[tr_idx], X_enc[val_idx]

        y_tr, y_vl = y.values[tr_idx], y.values[val_idx]

        

        yield X_tr, y_tr, X_vl, y_vl, X_test_enc

        

        del X_tr, X_vl, y_tr, y_vl, tr_idx, val_idx

        gc.collect()

        

    del loc_encoder, X_enc, X_test_enc

    gc.collect()





def single_validation(encoder, X, y, X_test, epochs):

    cv = KFold(n_splits=epochs, shuffle=True, random_state=SEED)



    for tr_idx, val_idx in cv.split(X.values, y.values): 

        X_tr, X_vl = X.values[tr_idx], X.values[val_idx]

        y_tr, y_vl = y.values[tr_idx], y.values[val_idx]



        loc_encoder = clone(encoder)

        loc_encoder.fit(X_tr, y_tr)

        

        yield loc_encoder.transform(X_tr), y_tr, loc_encoder.transform(X_vl), y_vl, loc_encoder.transform(X_test.values)

        

        del loc_encoder, X_tr, X_vl, y_tr, y_vl

        gc.collect()

        

        

def double_validation(encoder, X, y, X_test, epochs):

    cv = KFold(n_splits=epochs, shuffle=True, random_state=SEED)



    for tr_idx, val_idx in cv.split(X.values, y.values): 

        X_tr, X_vl = X.values[tr_idx], X.values[val_idx]

        y_tr, y_vl = y.values[tr_idx], y.values[val_idx]

        

        X_tr_enc = np.zeros(X_tr.shape)

        X_vl_enc = np.zeros(X_vl.shape)

        X_test_enc = np.zeros(X_test.shape)

        

        for sub_tr_idx, sub_val_idx in cv.split(X_tr, y_tr): 

            

            sub_X_tr, sub_X_vl = X_tr[sub_tr_idx], X_tr[sub_val_idx]

            sub_y_tr, sub_y_vl = y_tr[sub_tr_idx], y_tr[sub_val_idx]

                        

            loc_encoder = clone(encoder)

            loc_encoder.fit_transform(sub_X_tr, sub_y_tr)

                                                        

            _X_test_enc = loc_encoder.transform(X_test.values)            

            if X_test_enc is None:

                X_test_enc = np.zeros(_X_test_enc.shape)

            X_test_enc += _X_test_enc

            

            _X_vl_enc = loc_encoder.transform(X_vl)            

            if X_vl_enc is None:

                X_vl_enc = np.zeros(_X_vl_enc.shape)

            X_vl_enc += _X_vl_enc     

            

            X_tr_enc[sub_val_idx] += loc_encoder.transform(sub_X_vl)

            

            del loc_encoder, sub_tr_idx, sub_val_idx, sub_X_tr, sub_X_vl,sub_y_tr, sub_y_vl, _X_test_enc, _X_vl_enc

            gc.collect()

        

        yield X_tr_enc, y_tr, X_vl_enc / epochs, y_vl, X_test_enc / epochs

        

        del X_tr, X_vl, y_tr, y_vl, X_tr_enc, X_vl_enc, X_test_enc, tr_idx, val_idx

        gc.collect()



    

def get_solution(X_train, y_train, X_test, encoder, validation, verbose=False):  

    def solution(params):

        EPOCHS = params['folds']        

        y_preds = np.zeros(X_test.shape[0])

       

        for bag, (X_tr, y_tr, X_vl, y_vl, X_test_enc) in enumerate(validation(encoder, X_train, y_train, X_test, EPOCHS)): 

            model = XGBClassifier(**params)                             

            model.fit(X_tr, y_tr)



            if verbose:                    

                y_pred_train = model.predict_proba(X_vl)[:,1]

                print('[{}] ROC AUC {}'.format(bag + 1, roc_auc_score(y_vl, y_pred_train)))

                del y_pred_train



            del X_tr, X_vl, y_tr, y_vl

            gc.collect()



            y_preds += model.predict_proba(X_test_enc)[:,1]

            del model, X_test_enc

            gc.collect()  



        return y_preds / EPOCHS

    

    return solution
single_val_encoders = [    

    ce.CatBoostEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan', random_state=SEED),

    ce.JamesSteinEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan', random_state=SEED),

    ce.LeaveOneOutEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan', random_state=SEED),

    ce.MEstimateEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan', random_state=SEED),

    ce.OrdinalEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan'),

    ce.TargetEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan'),

    ce.WOEEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan', random_state=SEED)

]



multi_val_encoders = [

#     ce.BackwardDifferenceEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan'),

#     ce.BaseNEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan'),

    ce.BinaryEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan'),

    ce.HashingEncoder(return_df=False),

#     ce.HelmertEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan'),

#     ce.OneHotEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan'),

#     ce.SumEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan'),

#     ce.PolynomialEncoder(return_df=False, handle_unknown='return_nan', handle_missing='return_nan')

]



encoders = single_val_encoders + multi_val_encoders
scores = pd.DataFrame(columns=['None Validation', 'Single Validation', 'Double Validation'])
params = {

    'folds': 3,

    'n_estimators': 500,

    'max_depth': 9,

    'learning_rate': 0.05,        

    'tree_method': 'gpu_hist',

    'random_state': SEED

}
for enc in encoders:

    

    start = time.perf_counter()

    

    y_val_pred = get_solution(X_train_sub, y_train, X_val, encoder=enc, validation=none_validation, verbose=True)(params)



    print(enc.__class__.__name__, roc_auc_score(y_val, y_val_pred), time.perf_counter() - start)   

    scores.loc[enc.__class__.__name__, 'None Validation'] = roc_auc_score(y_val, y_val_pred)
for enc in encoders:

    

    start = time.perf_counter()

    

    y_val_pred = get_solution(X_train_sub, y_train, X_val, encoder=enc, validation=single_validation, verbose=True)(params)



    print(enc.__class__.__name__, roc_auc_score(y_val, y_val_pred), time.perf_counter() - start)   

    scores.loc[enc.__class__.__name__, 'Single Validation'] = roc_auc_score(y_val, y_val_pred)
for enc in single_val_encoders:

    

    start = time.perf_counter()

    

    y_val_pred = get_solution(X_train_sub, y_train, X_val, encoder=enc, validation=double_validation, verbose=True)(params)



    print(enc.__class__.__name__, roc_auc_score(y_val, y_val_pred), time.perf_counter() - start)  

    scores.loc[enc.__class__.__name__, 'Double Validation'] = roc_auc_score(y_val, y_val_pred)
scores