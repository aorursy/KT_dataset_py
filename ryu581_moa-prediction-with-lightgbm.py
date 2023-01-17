# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import library

import numpy as np

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb



from tqdm import tqdm
# import datasets



X_test = pd.read_csv('../input/lish-moa/test_features.csv')

X_train = pd.read_csv('../input/lish-moa/train_features.csv')

y_train = pd.read_csv('../input/lish-moa/train_targets_scored.csv')



sample = pd.read_csv('../input/lish-moa/sample_submission.csv')

nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
# drop useless column

X_test = X_test.drop('sig_id', axis=1)

X_train = X_train.drop('sig_id', axis=1)

y_train = y_train.drop('sig_id', axis=1)
# feature engineering

# hot encoding on cp_type, cp_time, cp_dose

categorical_features = ['cp_type', 'cp_time', 'cp_dose']

for column in categorical_features:

    X_train[column] = X_train[column].astype('category')

    X_train[column] = X_train[column].cat.codes.astype('int16')

    X_test[column] = X_test[column].astype('category')

    X_test[column] = X_test[column].cat.codes.astype('int16')
predict_labels = list(y_train.columns)
# prediction phase

result_score = pd.DataFrame()



for label in tqdm(predict_labels):

    print('start training model for {}'.format(label))

    train_label_df = y_train[label]



    # create empty list to store prediction models

    models = []

    row_no_lists = list(range(len(train_label_df)))



    # create K_fold instance (in this case K=5)

    K_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)



    for train_cv_no, eval_cv_no in K_fold.split(row_no_lists, train_label_df):

        X_train_cv = X_train.iloc[train_cv_no, :]

        y_train_cv = train_label_df[train_cv_no]

        X_eval_cv = X_train.iloc[eval_cv_no, :]

        y_eval_cv = train_label_df[eval_cv_no]



        lgb_train = lgb.Dataset(X_train_cv, y_train_cv, categorical_feature=categorical_features, free_raw_data=False)

        lgb_eval = lgb.Dataset(X_eval_cv, y_eval_cv, reference=lgb_train, categorical_feature=categorical_features, free_raw_data=False)



        params = {

            'task':'train',

            'boosting_type':'gbdt',

            'objective':'binary',

            'metric':'binary_logloss',

            # 'num_class':1,

            'learning_rate':0.05,

            'num_leaves':25,

            'min_data_in_leaf':25,

            #'device_type':'gpu',

            'bagging_fraction':1.0,

            'bagging_freq':0,

            'feature_fraction':1.0,

            'max_bin':127,

            'max_depth':5,

            'save_binary':True

        }



        evaluation_results = {}

        model = lgb.train(

            params, lgb_train, num_boost_round=3000, valid_names=['train', 'valid'], valid_sets=[lgb_train, lgb_eval], evals_result=evaluation_results, categorical_feature=categorical_features, early_stopping_rounds=500, verbose_eval=100

        )

        models.append(model)



    print('end of model training for {}'.format(label))

    print()



    result_df = pd.DataFrame()

    for idx, model in enumerate(models):

        result_df['fold_{}'.format(idx+1)] = model.predict(X_test, num_iteration=model.best_iteration)

    result_score[label] = result_df.mean(axis=1)
result_score
submission = pd.concat([sample['sig_id'], result_score], axis=1)
submission.to_csv('submission.csv', index=False)
submission