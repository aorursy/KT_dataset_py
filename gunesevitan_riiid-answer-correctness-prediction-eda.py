!pip install ../input/riiid-answer-correctness-prediction-utilities/site-packages/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl # datatable-0.11.0
import warnings

warnings.filterwarnings('ignore')



from datetime import datetime



import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import datatable as dt



from numba import jit

from sklearn.metrics import roc_auc_score
TRAIN_DTYPES = {

    'row_id': np.uint32,

    'timestamp': np.uint64,

    'user_id': np.uint32,

    'content_id': np.uint16,

    'content_type_id': np.uint8,

    'task_container_id': np.uint16,

    'user_answer': np.int8,

    'answered_correctly': np.int8,

    'prior_question_elapsed_time': np.float32,

    'prior_question_had_explanation': 'boolean'

}



df_train = dt.fread('../input/riiid-test-answer-prediction/train.csv').to_pandas()



for column, dtype in TRAIN_DTYPES.items():

    df_train[column] = df_train[column].astype(dtype) 

    

df_train['prior_question_had_explanation'] = df_train['prior_question_had_explanation'].astype(np.float16).fillna(-1).astype(np.int8)



print(f'Training Set Shape = {df_train.shape}')

print(f'Training Set Memory Usage = {df_train.memory_usage().sum() >> 20:.2f} MB')
QUESTIONS_DTYPES = {

    'question_id': np.uint16,

    'bundle_id': np.uint16,

    'correct_answer': np.uint8,

    'part': np.uint8,

}



df_questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv', dtype=QUESTIONS_DTYPES)



print(f'Questions Shape = {df_questions.shape}')

print(f'Questions Memory Usage = {df_questions.memory_usage().sum() >> 20:.2f} MB')
LECTURES_DTYPES = {

    'lecture_id': np.uint16,

    'tag': np.uint16,

    'part': np.uint8

}



df_lectures = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv', dtype=LECTURES_DTYPES)

df_lectures['type_of'] = np.uint8(df_lectures['type_of'].map({'concept': 0, 'solving question': 1, 'intention': 2, 'starter': 3}))



print(f'Lectures Shape = {df_lectures.shape}')

print(f'Lectures Memory Usage = {df_lectures.memory_usage().sum() >> 20:.2f} MB')
df_example_test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')



print(f'Example Test Set Shape = {df_example_test.shape}')

print(f'Example Test Set Memory Usage = {df_example_test.memory_usage().sum() >> 20:.2f} MB')
import riiideducation



env = riiideducation.make_env()

env
iter_test = env.iter_test()

iter_test
df_group, df_group_submission = next(iter_test)



print(f'Test Set Group\n{"-" * 14}\n\n{df_group}\n\n')

print(f'Submission\n{"-" * 10}\n\n{df_group_submission}')
df_group['answered_correctly'] = 0.5

env.predict(df_group.loc[df_group['content_type_id'] == 0, ['row_id', 'answered_correctly']])



for df_group, df_group_submission in iter_test:

    df_group['answered_correctly'] = 0.5

    env.predict(df_group.loc[df_group['content_type_id'] == 0, ['row_id', 'answered_correctly']])
@jit

def fast_roc_auc_score(y_true, y_prob):

    

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    

    n_false = 0

    auc = 0

    n = len(y_true)

    

    for i in range(n):

        y_i = y_true[i]

        n_false += (1 - y_i)

        auc += y_i * n_false

    auc /= (n_false * (n - n_false))

    return auc



questions = df_train['answered_correctly'] != -1



start = datetime.now()

fast_roc_auc_score(df_train.loc[questions, 'answered_correctly'], df_train.loc[questions, 'answered_correctly'])

fast_roc_auc_time = (datetime.now() - start).total_seconds()

print(f'fast_roc_auc_score finished in {fast_roc_auc_time:.2} seconds.')



start = datetime.now()

roc_auc_score(df_train.loc[questions, 'answered_correctly'], df_train.loc[questions, 'answered_correctly'])

roc_auc_score_time = (datetime.now() - start).total_seconds()

print(f'sklearn.metrics.roc_auc_score finished in {roc_auc_score_time:.4} seconds.')



del questions