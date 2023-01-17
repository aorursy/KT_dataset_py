import riiideducation

import pandas as pd



# You can only call make_env() once, so don't lose it!

env = riiideducation.make_env()
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=10**5, 

                       dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

                             'prior_question_had_explanation': 'boolean',

                             }

                      )
train_df['answered_correctly'].value_counts()
content_acc = train_df.query('answered_correctly != -1').groupby('content_id')['answered_correctly'].mean().to_dict()
iter_test = env.iter_test()
def add_content_acc(x):

    if x in content_acc.keys():

        return content_acc[x]

    else:

        return 0.5





for (test_df, sample_prediction_df) in iter_test:

    test_df['answered_correctly'] = test_df['content_id'].apply(add_content_acc).values

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])