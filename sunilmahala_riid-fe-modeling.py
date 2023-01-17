import optuna

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostClassifier

from  sklearn.tree import DecisionTreeClassifier

from  sklearn.model_selection import train_test_split

import operator

import random



# visualize

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

from matplotlib import pyplot

from matplotlib.ticker import ScalarFormatter

sns.set_context("talk")

style.use('fivethirtyeight')

import matplotlib.pyplot as plt

import seaborn as sns

import riiideducation

import dask.dataframe as dd

import  pandas as pd

import numpy as np

from sklearn.metrics import roc_auc_score

import riiideducation



env = riiideducation.make_env()

data= pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                nrows=10**7, dtype={'timestamp': 'int64', 'user_id': 'int32' ,

                                                  'content_id': 'int16','content_type_id': 'int8',

                                    'task_container_id':'int16','user_answer':'int8',

                                                  'answered_correctly':'int8',

                                                  'prior_question_elapsed_time': 'float32',

                                                  'prior_question_had_explanation': 'boolean'}

              )
print(data.shape)

data.head()
## removing letures data

data = data[data['content_type_id']==0]

print(data.shape)
## sort by timestamp 

data = data.sort_values(['user_id','timestamp'])

data.head()
user_df = data[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'sum', 'count'])

user_df.columns = ['answered_correctly_user', 'sum_user', 'count_user']

user_df.head()
content_df = data[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean', 'sum', 'count'])

content_df.columns = ['answered_correctly_content', 'sum_content', 'count_content']

content_df.head()
#reading in question df

questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv',

#                             usecols=[0, 3],

#                             dtype={'question_id': 'int16',

#                               'part': 'int8'}

                          )
questions_df
questions_df['part'].value_counts()
questions_df = questions_df.merge(content_df,

                                  left_on = 'question_id', right_on = 'content_id', how = 'left')

questions_df
bundle_df = questions_df.groupby('bundle_id')

bundle_df = bundle_df.agg({'sum_content': 'sum', 'count_content': 'sum'}).copy()

bundle_df.columns = ['bundle_rignt_answers', 'bundle_questions_asked']

bundle_df['bundle_accuracy'] = bundle_df['bundle_rignt_answers'] / bundle_df['bundle_questions_asked']

bundle_df
part_df = questions_df.groupby('part')

part_df = part_df.agg({'sum_content': 'sum', 'count_content': 'sum'}).copy()

part_df.columns = ['part_rignt_answers', 'part_questions_asked']

part_df['part_accuracy'] = part_df['part_rignt_answers'] / part_df['part_questions_asked']

part_df
data = data.merge(user_df, how = 'left', on = 'user_id')

data = data.merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')

data = data.merge(bundle_df, how = 'left', on = 'bundle_id')

data = data.merge(part_df, how = 'left', on = 'part')

data.head()
data.columns[data.isna().any()].tolist()
data['prior_question_elapsed_time'].fillna(data.groupby('user_id')

                                           ['prior_question_elapsed_time'].transform('mean'),inplace=True)

data['prior_question_elapsed_time'].fillna(data['prior_question_elapsed_time'].mean(),inplace=True)
data['prior_question_had_explanation'].fillna(data['prior_question_had_explanation'].mode()[0],inplace=True)
data.columns[data.isna().any()].tolist()
data.head()
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()



data["prior_question_had_explanation"] = lb_make.fit_transform(data["prior_question_had_explanation"])

data.head()
##  creating validation set

validation = pd.DataFrame()

for i in range(15):

    last_records = data.drop_duplicates('user_id', keep = 'last')

    data = data[~data.index.isin(last_records.index)]

    validation = validation.append(last_records)

print(len(data) , len(validation))
# X = pd.DataFrame()

# for i in range(15):

#     last_records = data.drop_duplicates('user_id', keep = 'last')

#     data = data[~data.index.isin(last_records.index)]

#     X = X.append(last_records)

# print(len(data),len(X))



X = data.sample(n=2000000)

print(len(data),len(X))
print(data.answered_correctly.mean() , X.answered_correctly.mean())
id(data),id(X)
del data 

import gc

gc.collect()
X.head()
y = X['answered_correctly']

X = X.drop(['answered_correctly'], axis=1)



y_val = validation['answered_correctly']

X_val = validation.drop(['answered_correctly'], axis=1)
features = ['timestamp', 'prior_question_elapsed_time','prior_question_had_explanation',

            'answered_correctly_user', 'sum_user', 'count_user', 'part','answered_correctly_content',

       'sum_content', 'count_content', 'bundle_rignt_answers',

       'bundle_questions_asked', 'bundle_accuracy', 'part_rignt_answers',

            'part_questions_asked', 'part_accuracy']





#features = ['answered_correctly_user', 'answered_correctly_content', 'sum_user', 'count_user',

  #     'prior_question_elapsed_time','prior_question_had_explanation', 'part']
X = X[features]

X_val = X_val[features]
X.head()
import lightgbm as lgb



params = {

    'objective': 'binary',

    'max_bin': 700,

    'learning_rate': 0.0175,

    'num_leaves': 80,

    'metric':'auc'

}



lgb_train = lgb.Dataset(X, y, categorical_feature = ['part', 'prior_question_had_explanation'])

lgb_eval = lgb.Dataset(X_val, y_val, categorical_feature = ['part', 'prior_question_had_explanation'], reference=lgb_train)



model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=50,

    num_boost_round=10000,

    early_stopping_rounds=12

)



y_pred = model.predict(X_val)

y_true = np.array(y_val)

roc_auc_score(y_true, y_pred)
# import lightgbm as lgb



# params = {

#     'objective': 'binary',

#     'max_bin': 700,

#     'learning_rate': 0.0175,

#     'num_leaves': 80,

#     'metric':'auc'

# }



# lgb_train = lgb.Dataset(X, y, categorical_feature = ['part', 'prior_question_had_explanation'])

# lgb_eval = lgb.Dataset(X_val, y_val, categorical_feature = ['part', 'prior_question_had_explanation'], reference=lgb_train)



# model = lgb.train(

#     params, lgb_train,

#     valid_sets=[lgb_train, lgb_eval],

#     verbose_eval=50,

#     num_boost_round=10000,

#     early_stopping_rounds=12

# )



# y_pred = model.predict(X_val)

# y_true = np.array(y_val)

# roc_auc_score(y_true, y_pred)
import matplotlib.pyplot as plt

import seaborn as sns
lgb.plot_importance(model)

plt.show()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    test_df = test_df.merge(user_df, how = 'left', on = 'user_id')

    test_df = test_df.merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')

    test_df = test_df.merge(bundle_df, how = 'left', on = 'bundle_id')

    test_df = test_df.merge(part_df, how = 'left', on = 'part')

    

    test_df['prior_question_elapsed_time'].fillna(test_df.groupby('user_id')

                                           ['prior_question_elapsed_time'].transform('mean'),inplace=True)

    test_df['prior_question_elapsed_time'].fillna(test_df['prior_question_elapsed_time'].mean(),inplace=True)

    test_df['prior_question_had_explanation'].fillna(test_df['prior_question_had_explanation'].mode()[0],inplace=True)

    test_df.fillna(value = -1, inplace = True)

    

    test_df["prior_question_had_explanation"] = lb_make.fit_transform(test_df["prior_question_had_explanation"])



    test_df['answered_correctly'] =  model.predict(test_df[features])

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])