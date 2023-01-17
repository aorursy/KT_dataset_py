from pandas import read_csv
train     = read_csv('../input/riiid-test-answer-prediction/train.csv',

                dtype={

                    'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

                    'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8',

                    'prior_question_elapsed_time': 'float32', 'prior_question_had_explanation': 'boolean',

                }

            )

questions = read_csv('../input/riiid-test-answer-prediction/questions.csv',

                dtype={'question_id': 'int16', 'bundle_id': 'int16', 'correct_answer' :'int8', 'part': 'int8', 'tags':'object'}

            )

lectures  = read_csv('../input/riiid-test-answer-prediction/lectures.csv',

                dtype={'lecture_id': 'int16', 'tag': 'int16', 'part': 'int8', 'type_of': 'category'}

            )
train.to_feather('train.feather')

questions.to_feather('questions.feather')

lectures.to_feather('lectures.feather')
%%writefile riiid_loader.py



def load_files():

    import pandas as pd

    import time

    start_time = time.time()

    path = '/kaggle/input/riiid-dataset-loader/'

    

    train = pd.read_feather(path+'train.feather')

    

    questions = pd.read_feather(path+'questions.feather')

    questions.tags.fillna('-1',inplace=True)

    questions.tags=[[int(y) for y in x if y !='-1'] for x in questions.tags.str.split()]

    

    lectures = pd.read_feather(path+'lectures.feather')

    lectures.type_of = lectures.type_of.cat.codes.astype('int8')

    

    print("--- Datasests loaded in %s seconds ---" % round(time.time() - start_time,3))

    return train,questions,lectures