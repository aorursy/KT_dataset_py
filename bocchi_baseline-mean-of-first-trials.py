import riiideducation

import pandas as pd



path_train_csv = '../input/riiid-test-answer-prediction/train.csv'

path_questions_csv = '../input/riiid-test-answer-prediction/questions.csv'



d_types = {'row_id': 'int64', #  0

           'timestamp': 'int64',#  1

           'user_id': 'int32', #  2

           'content_id': 'int16',#  3

           'content_type_id': 'int8',#  4

           'task_container_id': 'int16',#  5

           'user_answer': 'int8', #  6

           'answered_correctly': 'int8',#  7

           'prior_question_elapsed_time': 'float32',#  8 

           'prior_question_had_explanation': 'boolean',#  9

         }
# read csv

train_df = pd.read_csv(path_train_csv,

                       dtype=d_types,

                       usecols=[2, 3, 4, 6]

                      )

target_df = train_df.query('content_type_id == 0')[['content_id', 'user_id', 'user_answer']]

del train_df

# obtain first trials

user_answer_df = target_df.groupby(['content_id', 'user_id']).first().user_answer

del target_df

# grouped by questions and obtain user reactions on their first trials

answer_hist_df = user_answer_df.groupby('content_id').value_counts()

del user_answer_df

answer_hist_df_ = answer_hist_df.unstack()

del answer_hist_df

answer_hist_df = answer_hist_df_.fillna(0).astype('int32')

answer_hist_df['total'] = answer_hist_df.sum(axis=1)
# join question data and culc correct rates

q_data_df = pd.read_csv(path_questions_csv)

q_df = pd.merge(q_data_df, answer_hist_df, left_on='question_id', right_on='content_id')

q_df['answered_correctly'] = q_df.apply(lambda x : x[x['correct_answer']] / x['total'], axis=1)
# put correct rate values as predicted 

env = riiideducation.make_env()

iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:

    t_test_df = pd.merge(test_df.query('content_type_id == 0'), q_df, right_on='question_id', left_on='content_id')[['row_id', 'answered_correctly']]

    env.predict(t_test_df)