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
#Example submission and test

sample_prediction_df_dtype = {'row_id': 'int64', 'answered_correctly': 'float16','group_num': 'int64'}

test_df_dtype = {'row_id': 'int64','group_num': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

            'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

            'prior_question_had_explanation': 'boolean'

                     }

train_dtype= {'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

            'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

            'prior_question_had_explanation': 'boolean'

            }

questions_dtype = {'question_id': 'int16','bundle_id': 'int16','correct_answer': 'int8','part': 'int8','tags':str}

lectures_dtype = {'lecture_id': 'int16','tags':str,'part': 'int8','type_of':str}



sample_prediction_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_sample_submission.csv',dtype=sample_prediction_df_dtype, low_memory=False)

test_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv',dtype=test_df_dtype, low_memory=False)

train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',dtype=train_dtype, low_memory=False, nrows=10**5,skiprows=range(1, 0))

questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv',dtype=questions_dtype, low_memory=False)

lectures = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv',dtype=lectures_dtype, low_memory=False)



print("DONE LOADING DATA.")
from sklearn.model_selection import train_test_split

from statsmodels.stats.proportion import proportion_confint



def transform_pqhe(df):

    df['prior_question_had_explanation'].fillna(True,inplace=True)

    df['prior_question_had_explanation']= df['prior_question_had_explanation'].apply(lambda x: int(x))

    return df



train = transform_pqhe(train)

train = train[train['content_type_id'] == 0]
train_df, val_df = train_test_split(train, test_size = 0.2, shuffle=True)
import lightgbm as lgb



features = ['prior_question_had_explanation'

           ,'answered_correctly_perc_question','answered_correctly_perc_question_lower_bound_95%','answered_correctly_perc_question_upper_bound_95%','upper_bound_diff_question','lower_bound_diff_question'

           ,'answered_correctly_perc_user', 'answered_correctly_perc_user_lower_bound_95%','answered_correctly_perc_user_upper_bound_95%','upper_bound_diff_user','lower_bound_diff_user']

features = ['prior_question_had_explanation']

target = ['answered_correctly']





lgb_train = lgb.Dataset(train_df[features], train_df[target])

lgb_eval = lgb.Dataset(val_df[features], val_df[target])



param = {'num_leaves':5,

         'max_depth':5,

         'num_leaves':5,

         'n_estimators ':5,

         'max_bin':5,

         'num_leaves':5,

         'reg_lambda':1,

         'metric': 'auc',

         'objective':'binary'}



# Train the model

clf = lgb.train(param,

                lgb_train,

                valid_sets=[lgb_train, lgb_eval],

                verbose_eval=5000,

                early_stopping_rounds=100)

predictions = clf.predict(val_df[features])

predictions
import riiideducation



env = riiideducation.make_env()



iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:

    #preprocessing 'pipeline'

    test_df = transform_pqhe(test_df)

    #add column with prediction

    test_df['answered_correctly'] = clf.predict(test_df[features])

#    test_df['answered_correctly'].fillna(0.5)

    #add prediction to environment

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])