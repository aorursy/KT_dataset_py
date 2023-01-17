!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl
import riiideducation

import pandas as pd

import numpy as np

import datatable as dt
# saving the dataset in .jay (binary format)

dt.fread("../input/riiid-test-answer-prediction/train.csv").to_jay("train.jay")
%%time



# reading the dataset from .jay format

import datatable as dt



train = dt.fread("train.jay")



print(train.shape)
%%time



train = train.to_pandas()[['content_id','prior_question_had_explanation','answered_correctly']]
train.head()
train.shape
## Answer df is the train data with only info about answers i.e containing no lectures

answer_df = train.query('answered_correctly != -1')

del train
answer_df.describe()
answer_df[['content_id','answered_correctly']].groupby(['content_id']).agg('mean').plot()
answer_df[['prior_question_had_explanation','answered_correctly']].groupby(['prior_question_had_explanation']).agg('mean').plot(kind='bar')
average_student_performance = answer_df.describe()['answered_correctly'][1]

average_student_performance
%%time

## Calculate accuracy by content and explanation

c_exp_acc = answer_df[['content_id','prior_question_had_explanation','answered_correctly']].groupby(['content_id','prior_question_had_explanation']).agg('mean').reset_index()

c_exp_acc.columns = ['content_id','prior_question_had_explanation', 'content_explanation_acc']

c_exp_acc.head()

del answer_df
# You can only call make_env() once, so don't lose it!

env = riiideducation.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:



    ## Add the calculated heuristics by content and question explanation

    ## Then fill the missing values by the average question answer.

    test_df = test_df.merge(c_exp_acc, how = 'left', on = ['content_id','prior_question_had_explanation'])

    test_df['answered_correctly'] = test_df['content_explanation_acc'].fillna(average_student_performance)

    



    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])