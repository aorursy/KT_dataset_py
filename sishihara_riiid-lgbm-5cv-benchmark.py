import lightgbm as lgb

import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score



import riiideducation
# You can only call make_env() once, so don't lose it!

env = riiideducation.make_env()
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=10**5, 

                       dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

                             'prior_question_had_explanation': 'boolean',

                             }

                      )

train_df = train_df.query('answered_correctly != -1').reset_index(drop=True)

train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].astype(float)
train_df.head()
y_train = train_df['answered_correctly']

X_train = train_df.drop(['answered_correctly', 'user_answer'], axis=1)
models = []

oof_train = np.zeros((len(X_train),))

cv = KFold(n_splits=5, shuffle=True, random_state=0)



categorical_features = ['user_id', 'content_type_id', 'task_container_id', 'prior_question_had_explanation']



params = {

    'objective': 'binary',

    'max_bin': 300,

    'learning_rate': 0.05,

    'num_leaves': 40

}



for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)



    model = lgb.train(

        params, lgb_train,

        valid_sets=[lgb_train, lgb_eval],

        verbose_eval=10,

        num_boost_round=1000,

        early_stopping_rounds=10

    )



    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)

    models.append(model)
roc_auc_score(y_train, oof_train)
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    y_preds = []

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].astype(float)

    X_test = test_df.drop(['prior_group_answers_correct', 'prior_group_responses'], axis=1)



    for model in models:

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        y_preds.append(y_pred)



    y_preds = sum(y_preds) / len(y_preds)

    test_df['answered_correctly'] = y_preds

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])