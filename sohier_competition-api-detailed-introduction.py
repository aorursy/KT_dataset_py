import riiideducation

import pandas as pd



# You can only call make_env() once, so don't lose it!

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
# You can only iterate through a result from `env.iter_test()` once

# so be careful not to lose it once you start iterating.

iter_test = env.iter_test()
(test_df, sample_prediction_df) = next(iter_test)

test_df
sample_prediction_df
next(iter_test)
env.predict(sample_prediction_df)
for (test_df, sample_prediction_df) in iter_test:

    test_df['answered_correctly'] = 0.5

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])