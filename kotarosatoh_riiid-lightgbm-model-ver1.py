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
dtypes = {
    'row_id': 'int64', 'timestamp': 'int64', 
    'user_id': 'int32', 'content_id': 'int16',
    'content_type_id': 'int8', 'task_container_id': 'int16',
    'user_answer': 'int8', 'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32', 'prior_question_had_explanation': 'boolean',
}

train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',
                    low_memory=False,
                    nrows=10**7, 
                    dtype=dtypes)
train.head()
# 問題の正解率
question_correctly = train.groupby('content_id').agg({'answered_correctly': 'mean'}).reset_index()
question_correctly = question_correctly.rename(columns={'answered_correctly': 'question_correctly'})
question_correctly.head()
# 問題ごとの解答数
question_answered = train.groupby('content_id').agg({'row_id': 'nunique'}).reset_index()
question_answered = question_answered.rename(columns={'row_id': 'question_answered'})
question_answered.head()
# 問題に正解したユーザーの数
question_correct_users = train[train.answered_correctly==1].groupby('content_id')\
                                                            .agg({'user_id': 'nunique'})\
                                                            .reset_index()
question_correct_users = question_correct_users.rename(columns={'user_id': 'question_correct_users'})
question_correct_users.head()
task_questions = train.groupby('task_container_id').agg({'content_id': 'nunique'}).reset_index()
task_questions = task_questions.rename(columns={'content_id': 'task_questions'})
task_questions.head()
train.timestamp
task_correct_time = train[(train.content_type_id==0)&(train.answered_correctly==1)]\
                        [['task_container_id', 'timestamp']]
task_correct_time = task_correct_time.groupby('task_container_id', as_index=False)\
                                    .agg({'timestamp': 'mean'})\
                                    .rename(columns={'timestamp': 'task_correct_time'})
task_correct_time.head()
train = pd.merge(train, question_correctly, on='content_id', how='left')
train = pd.merge(train, question_answered, on='content_id', how='left')
train = pd.merge(train, question_correct_users, on='content_id', how='left')

train = pd.merge(train, task_questions, on='task_container_id', how='left')
train = pd.merge(train, task_correct_time, on='task_container_id', how='left')
train.head()
train['task_time_diff'] = train.timestamp - train.task_correct_time
train.head()
feature_columns = [
    'task_time_diff', 'prior_question_elapsed_time',
    'question_correctly', 'question_answered', 'question_correct_users',
    'task_questions', 
]
avg_elapsed_time = train[train.content_type_id==0].prior_question_elapsed_time.mean()
train.prior_question_elapsed_time = train.prior_question_elapsed_time.fillna(avg_elapsed_time)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# train = train[train.content_type_id==0]
# validation = train.groupby('user_id').tail(30)
# train = train[~train.index.isin(validation.index)]
# len(validation), len(train)
# X_train, y_train = train[feature_columns].values, train['answered_correctly'].values
# len(X_train), len(y_train)
# X_test, y_test = validation[feature_columns].values, validation['answered_correctly'].values
X = train[train.content_type_id==0][feature_columns].values
y = train[train.content_type_id==0].answered_correctly.values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=2)
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_pred_prob[:, 1])
print('AUC :', auc) 
import matplotlib.pyplot as plt

# 特徴量の重要度を含むデータフレームを作成
imp_df = pd.DataFrame()
imp_df["feature"] = feature_columns
imp_df["importance"] = model.feature_importances_
imp_df = imp_df.sort_values("importance")

# 可視化
plt.figure(figsize=(7, 10))
plt.barh(imp_df.feature, imp_df.importance)
plt.xlabel("Feature Importance")
plt.show()
question_only_train = train[train.content_type_id==0].copy()

avg_question_correctly = question_only_train.question_correctly.mean()
avg_question_answered = question_only_train.question_answered.mean()
avg_question_correct_users = question_only_train.question_correct_users.mean()
avg_task_questions = question_only_train.task_questions.mean()

avg_elapsed_time = question_only_train.prior_question_elapsed_time.mean()
avg_task_time_diff = question_only_train.task_time_diff.mean()
import riiideducation
env = riiideducation.make_env()

iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:
    test_df = pd.merge(test_df, question_correctly, on='content_id', how='left')
    test_df = pd.merge(test_df, question_answered, on='content_id', how='left')
    test_df = pd.merge(test_df, question_correct_users, on='content_id', how='left')
    test_df = pd.merge(test_df, task_questions, on='task_container_id', how='left')
    test_df = pd.merge(test_df, task_correct_time, on='task_container_id', how='left')
    test_df['task_time_diff'] = test_df.timestamp - test_df.task_correct_time
    
    test_df.question_correctly = test_df.question_correctly.fillna(avg_question_correctly)
    test_df.question_answered = test_df.question_answered.fillna(avg_question_answered)
    test_df.question_correct_users = test_df.question_correct_users.fillna(avg_question_correct_users)
    test_df.task_questions = test_df.task_questions.fillna(avg_task_questions)
    test_df.task_time_diff = test_df.task_time_diff.fillna(avg_task_time_diff)
    
    test_df.prior_question_elapsed_time = test_df.prior_question_elapsed_time.fillna(avg_elapsed_time)

    y_pred = model.predict_proba(test_df[feature_columns].values)
    test_df['answered_correctly'] = y_pred[:, 1]
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
