# Read test data. I have previously split the data into K-folds, and also joined with `users.csv`



# The K-splits are stored in this folder

DATA_FOLDER = '/kaggle/input/shopee-w8/kfolds'



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os





test = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'), parse_dates=['grass_date'])

test.head()
NFOLDS = 5





trains = []

valids = []

for fold_id in range(NFOLDS):

    trains.append(pd.read_csv(os.path.join(DATA_FOLDER, str(fold_id), 'train.csv'), parse_dates=['grass_date']))

    valids.append(pd.read_csv(os.path.join(DATA_FOLDER, str(fold_id), 'valid.csv'), parse_dates=['grass_date']))





trains[0].head()
# Fit OneHotEncoders



from sklearn.preprocessing import OneHotEncoder





CATEGORICAL_COLS = ['domain', 'country_code']



OH_encoders = [OneHotEncoder(handle_unknown='ignore') for i in range(NFOLDS)]

for i in range(NFOLDS):

    print('Processing fold', i)

    OH_encoders[i].fit(trains[i][CATEGORICAL_COLS])
# Build features





LAST_DAY_COLS = ['last_open_day', 'last_login_day', 'last_checkout_day']

DROP_COLS = ['user_id', 'row_id', 'subject_line_length']



X_trains = [None for i in range(NFOLDS)]

y_trains = [None for i in range(NFOLDS)]

X_valids = [None for i in range(NFOLDS)]

y_valids = [None for i in range(NFOLDS)]

X_tests = [None for i in range(NFOLDS)]





def convert_last(s):

    if s in ['Never open', 'Never checkout', 'Never login']:

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



    # Process one-hot columns

    OH_cols = pd.DataFrame(OH_encoders[fold_id].transform(res[CATEGORICAL_COLS]).toarray())

    OH_cols.index = res.index

    res.drop(columns=CATEGORICAL_COLS, inplace=True)

    res = pd.concat([res, OH_cols], axis=1)



    # Process columns with NA

    for col in ['attr_1', 'attr_2', 'age']:

        res[col].fillna(-1, inplace=True)



    return res, target





for fold_id in range(NFOLDS):

    print('Processing fold', fold_id)

    X_trains[fold_id], y_trains[fold_id] = build_features(fold_id, trains[fold_id])

    X_valids[fold_id], y_valids[fold_id] = build_features(fold_id, valids[fold_id])

    X_tests[fold_id], _ = build_features(fold_id, test)





X_trains[0].head()
from imblearn.pipeline import make_pipeline, Pipeline

from imblearn.over_sampling import SMOTE

import optuna

from sklearn.model_selection import cross_validate

from sklearn.metrics import matthews_corrcoef, make_scorer

from sklearn.ensemble import RandomForestClassifier
"""

Results:



===== Done fold 0 =====

0.5178508619346296

{'n_estimators': 817, 'max_depth': 38, 'min_samples_split': 26, 'min_samples_leaf': 8, 'smoth_n_neighbors': 9}



===== Done fold 1 =====

0.5109257631569604

{'n_estimators': 200, 'max_depth': 33, 'min_samples_split': 147, 'min_samples_leaf': 9, 'smoth_n_neighbors': 9}



===== Done fold 2 =====

0.5095566726766917

{'n_estimators': 705, 'max_depth': 29, 'min_samples_split': 4, 'min_samples_leaf': 43, 'smoth_n_neighbors': 8}



===== Done fold 3 =====

0.5130338493405462

{'n_estimators': 274, 'max_depth': 33, 'min_samples_split': 41, 'min_samples_leaf': 14, 'smoth_n_neighbors': 9}



===== Done fold 4 =====

0.5115307558584887

{'n_estimators': 613, 'max_depth': 18, 'min_samples_split': 70, 'min_samples_leaf': 18, 'smoth_n_neighbors': 6}



"""





studies = [None for i in range(NFOLDS)]





# Change following line to range(NFOLDS) to run & find best params.

for fold_id in range(NFOLDS, NFOLDS):

    print('========== Processing fold', fold_id, '==========')



    def objective(trial:optuna.trial.Trial):

        params = {

            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),

            'max_depth': trial.suggest_int('max_depth', 4, 50),

            'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),

            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),

        }



        smoth_n_neighbors = trial.suggest_int('smoth_n_neighbors', 5, 10)

        sampler = SMOTE(random_state=42, k_neighbors=smoth_n_neighbors)



        clf = RandomForestClassifier(random_state=42, **params)

        pipeline = make_pipeline(sampler, clf)

        scores = cross_validate(pipeline, X_trains[fold_id], y_trains[fold_id], verbose=1,

                    n_jobs=-1, scoring=make_scorer(matthews_corrcoef), cv=4)

        return scores["test_score"].mean()





    studies[fold_id] = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

    studies[fold_id].optimize(objective, n_trials=20)



    print(f'===== Done fold {fold_id} =====')

    print(studies[fold_id].best_value)

    print(studies[fold_id].best_params)
# Build the final models with best params found above.



rf_best_params = [

    {'n_estimators': 817, 'max_depth': 38, 'min_samples_split': 26, 'min_samples_leaf': 8},

    {'n_estimators': 200, 'max_depth': 33, 'min_samples_split': 147, 'min_samples_leaf': 9},

    {'n_estimators': 705, 'max_depth': 29, 'min_samples_split': 4, 'min_samples_leaf': 43},

    {'n_estimators': 274, 'max_depth': 33, 'min_samples_split': 41, 'min_samples_leaf': 14},

    {'n_estimators': 613, 'max_depth': 18, 'min_samples_split': 70, 'min_samples_leaf': 18},

]

smote_best_params = [

    {'smoth_n_neighbors': 9},

    {'smoth_n_neighbors': 9},

    {'smoth_n_neighbors': 8},

    {'smoth_n_neighbors': 9},

    {'smoth_n_neighbors': 6},

]



rfs = []

samplers = []

mcc_sum = 0.0

for i in range(NFOLDS):

    print('Processing fold', i)

    samplers.append(SMOTE(random_state=42, k_neighbors=smote_best_params[i]["smoth_n_neighbors"]))

    rf = RandomForestClassifier(random_state=42, **rf_best_params[i])

    rfs.append(rf)



    pipeline = Pipeline([("sampler", samplers[i]), ("clf", rf)])

    pipeline.fit(X_trains[i], y_trains[i])

    preds_valid = rf.predict(X_valids[i].values)

    mcc = matthews_corrcoef(y_valids[i], preds_valid)

    print('MCC:', mcc)

    mcc_sum += mcc





print('Avg MCC:', mcc_sum / NFOLDS)
from tqdm.notebook import tqdm





def print_result(clfs, Xs):

    probs = np.zeros(shape=(Xs[0].shape[0], 2))



    for fold_id in tqdm(range(len(clfs))):

        probs += clfs[fold_id].predict_proba(Xs[fold_id]) / NFOLDS

    preds = np.argmax(probs, axis=1)



    submission = pd.DataFrame({

        'row_id': test['row_id'],

        'open_flag': preds,

    })

    submission.to_csv('submission.csv', index=False)





print_result(rfs, X_tests)