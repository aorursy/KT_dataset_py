import numpy as np # MATH

import pandas as pd # EATS BAMBOO

import gc # TRASHY

train_dtypes = {

    'row_id': np.int64,

    'timestamp': np.int64,

    'user_id': np.int32,

    'content_id': np.int16,

    'content_type_id': 'boolean',

    'task_container_id': np.int16,

    'user_answer': np.int8,

    'answered_correctly': np.int8,

    'prior_question_elapsed_time': np.float32,

    #'prior_question_had_explanation': np.float32

}



lecture_dtypes = {

    'lecture_id': np.int32,

    'tag': np.int32,

    'part': np.int8

}



question_dtypes = {

    'question_id': np.int32,

    'bundle_id': np.int32,

    'correct_answer': np.int8,

    'part': np.int8,

}
%%time

lectures = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv', dtype=lecture_dtypes)

train = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/train.csv", dtype=train_dtypes)

questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv', dtype=question_dtypes)
train.head()
train.prior_question_had_explanation = train.prior_question_had_explanation.map({True: 1, False: 0}).astype(np.float32)

#train.prior_question_had_explanation.map({True: 1, False: 0}).astype('Int8')
train.to_feather('train.feather')

lectures.to_feather('lectures.feather')

questions.to_feather('questions.feather')
del train

del lectures

del questions

gc.collect()
%%time

train = pd.read_feather('train.feather')

lectures = pd.read_feather('lectures.feather')

questions = pd.read_feather('questions.feather')
train.info()
train