# I have previously split the data into K-folds. Also joined with users.csv.



DATA_FOLDER = '/kaggle/input/shopee-w8/kfolds'

!ls {DATA_FOLDER}
# Read test data



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os





test = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'), parse_dates=['grass_date'])

test.head()
# Read train & validation data



NFOLDS = 5





trains = []

valids = []

for fold_id in range(NFOLDS):

    trains.append(pd.read_csv(os.path.join(DATA_FOLDER, str(fold_id), 'train.csv'), parse_dates=['grass_date']))

    valids.append(pd.read_csv(os.path.join(DATA_FOLDER, str(fold_id), 'valid.csv'), parse_dates=['grass_date']))





trains[0].head()
# Build features

# Note: For CatBoost, we don't need to encode Categorical columns. CatBoost will automatically take care of it.





CATEGORICAL_COLS = ['country_code', 'attr_1', 'attr_2', 'attr_3', 'domain']

LAST_DAY_COLS = ['last_open_day', 'last_login_day', 'last_checkout_day']

DROP_COLS = ['user_id', 'row_id', 'subject_line_length']



X_trains = [None for i in range(NFOLDS)]

y_trains = [None for i in range(NFOLDS)]

X_valids = [None for i in range(NFOLDS)]

y_valids = [None for i in range(NFOLDS)]

X_tests = [None for i in range(NFOLDS)]





def convert_last(s):

    if s in ['Never open', 'Never checkout', 'Never login', 'grass_day_of_week']:

        return -1

    return int(s)





def build_features(fold_id, dataset):

    target = None

    res = dataset.drop(columns=['user_id', 'row_id', 'subject_line_length'])

    if 'open_flag' in dataset.columns:

        target = res['open_flag']

        res.drop(columns=['open_flag'], inplace=True)

    

    # Last day columns: convert to int

    for col in LAST_DAY_COLS:

        res[col] = res[col].apply(convert_last)

    

    # Grass date: convert to day of week

    res['grass_day_of_week'] = res['grass_date'].apply(lambda x: x.weekday())

    res.drop(columns=['grass_date'], inplace=True)



    # Process columns with NA

    for col in ['attr_1', 'attr_2', 'attr_3', 'age']:

        res[col].fillna(-1, inplace=True)

        res[col] = res[col].astype(int)



    return res, target





for fold_id in range(NFOLDS):

    print('Processing fold', fold_id)

    X_trains[fold_id], y_trains[fold_id] = build_features(fold_id, trains[fold_id])

    X_valids[fold_id], y_valids[fold_id] = build_features(fold_id, valids[fold_id])

    X_tests[fold_id], _ = build_features(fold_id, test)





X_trains[0].head()
# CatBoost modeling





import optuna

from sklearn.metrics import matthews_corrcoef

from catboost import CatBoostClassifier



params = {

    'eval_metric': 'MCC',

    'cat_features': CATEGORICAL_COLS,

    'verbose': 200,

    'random_seed': 42,

    'od_pval': 1e-2,

    'use_best_model': True,

}



cats = []

mcc_sum = 0.0

for i in range(NFOLDS):

    class_weights = {0: 1.0, 1: 2}

    clf = CatBoostClassifier(class_weights=class_weights, **params)



    clf.fit(

        X_trains[i], y_trains[i],

        eval_set=(X_valids[i], y_valids[i]),

        use_best_model=True,

        plot=True,

    )

    mcc = matthews_corrcoef(y_valids[i], clf.predict(X_valids[i]))

    cats.append(clf)

    print(f'========== Results for fold {i} ==========')

    print('MCC:', mcc)

    mcc_sum += mcc





print('Avg MCC:', mcc_sum / NFOLDS)
def print_result(clfs, Xs):

    probs = np.zeros(shape=(Xs[0].shape[0], 2))



    for fold_id in range(len(clfs)):

        probs += clfs[fold_id].predict_proba(Xs[fold_id]) / NFOLDS

    preds = np.argmax(probs, axis=1)



    submission = pd.DataFrame({

        'row_id': test['row_id'],

        'open_flag': preds,

    })

    submission.to_csv('submission.csv', index=False)





print_result(cats, X_tests)