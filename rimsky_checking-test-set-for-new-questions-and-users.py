import riiideducation

import pandas as pd



env = riiideducation.make_env()
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=10**5, 

                       dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

                             'prior_question_had_explanation': 'boolean',

                             }

                      )

train_df
users = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', sep=',', usecols=['user_id', 'timestamp'], squeeze=True)

#this takes few minutes - reading in the entire set of users
users_with_latest_ts = users.groupby('user_id')['timestamp'].max()

#get the latest timestamp for all train users
#create set for comparision to the test set

user_set = set(users.user_id.unique())

len(user_set)
questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')
questions_df.question_id.max() + 1 == questions_df.shape[0] 

questions_df.shape[0]
iter_test = env.iter_test()
(test_df, sample_prediction_df) = next(iter_test)

test_df
#get users and timestamps

test_users_and_ts = test_df[['user_id','timestamp']]

test_users_and_ts.shape
#work with sets to create a set of unique users and questions returned by test API

question_ids = set(test_df.content_id.unique())

new_ids = set(test_df.content_id.unique())

question_ids = question_ids.union(new_ids)



user_ids = set(test_df.user_id.unique())

new_users = set(test_df.user_id.unique())

user_ids = user_ids.union(new_users)
env.predict(sample_prediction_df)
for (test_df, sample_prediction_df) in iter_test:

    new_ids = set(test_df.content_id.unique())

    question_ids = question_ids.union(new_ids)

    

    new_users = set(test_df.user_id.unique())

    user_ids = user_ids.union(new_users)

    

    print("Length of test set {}, unique users {}".format(len(test_df), len(new_ids)))

    

    test_users_and_ts_i = test_df[['user_id','timestamp']]

    test_users_and_ts = pd.concat([test_users_and_ts,test_users_and_ts_i])

    #print(test_users_and_ts.shape)

    

    test_df['answered_correctly'] = 0.5

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
test_set_min_ts = test_users_and_ts.groupby('user_id')['timestamp'].min().reset_index()
df = pd.merge(users_with_latest_ts,test_set_min_ts, on = 'user_id')
if any(df['timestamp_y'] < df['timestamp_x']): 

    print("USER INTERACTION IN TEST SET HAS HAPPENED _BEFORE_ THE LATEST INTERACTION IN TRAIN SET. TIME MIXUP DETECTED")

else:

    print("ALL CLEAR, TEST SET ACTIONS FOLLOWED TRAIN SET ACTIONS FOR ANY GIVEN USER WHO WAS PRESENT IN BOTH")
print(user_ids - user_set, "these users are new")
print(question_ids - set(questions_df.question_id), "these questions are new")
new_users_are_really_new = test_users_and_ts[test_users_and_ts.user_id.isin(user_ids - user_set)].groupby('user_id')['timestamp'].min().reset_index()

if new_users_are_really_new.timestamp.max() > 0:

    print("new user detected in test who is not really new! (timestamp is not 0)")

    print(new_users_are_really_new[new_users_are_really_new.timestamp>0])

else:

    print("ALL CLEAR. NEW USERS IN TEST ARE INDEED NEW - test contains their first interaction and possibly more")
new_users_are_really_new