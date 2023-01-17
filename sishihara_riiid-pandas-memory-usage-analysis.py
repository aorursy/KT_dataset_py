import pandas as pd



import riiideducation
# You can only call make_env() once, so don't lose it!

env = riiideducation.make_env()
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                       low_memory=False,

                       nrows=10**6,

                       dtype={'row_id': 'int64',

                              'timestamp': 'int64',

                              'user_id': 'int32',

                              'content_id': 'int16',

                              'content_type_id': 'int8',

                              'task_container_id': 'int16',

                              'user_answer': 'int8',

                              'answered_correctly': 'int8',

                              'prior_question_elapsed_time': 'float32',

                              'prior_question_had_explanation': 'boolean',

                             }

                      )
train_df.info()
train_df.memory_usage()
train_df.memory_usage().plot.barh()
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                       low_memory=False,

                       nrows=10**6,

                       usecols=['content_id', 'answered_correctly'],

                       dtype={'content_id': 'int16', 'answered_correctly': 'int8'}

                      )
train_df.info()