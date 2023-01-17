import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go



import sys

import gc
reader = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv', 

    low_memory=False, 

    chunksize=101230332 / 4, 

    dtype={

        'row_id': 'int64', 

        'timestamp': 'int64', 

        'user_id': 'int32', 

        'content_id': 'int16', 

        'content_type_id': 'int8',

        'task_container_id': 'int16', 

        'user_answer': 'int8', 

        'answered_correctly': 'int8',

        'prior_question_elapsed_time': 'float32', 

        'prior_question_had_explanation': 'boolean'

    }

)



reader
pd.options.display.float_format = '{:.11g}'.format

train_df = reader.get_chunk(101230332 / 4)

train_df.describe()
del train_df

gc.collect()
train_df = reader.get_chunk(101230332 / 4)

train_df.describe()
del train_df

gc.collect()
train_df = reader.get_chunk(101230332 / 4)

train_df.describe()
del train_df

gc.collect()
train_df = reader.get_chunk(101230332 / 4)

train_df.describe()
del train_df

gc.collect()
reader = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv', 

    low_memory=False, 

    chunksize=101230332 / 2, 

    dtype={

        'row_id': 'int64', 

        'timestamp': 'int64', 

        'user_id': 'int32', 

        'content_id': 'int16', 

        'content_type_id': 'int8',

        'task_container_id': 'int16', 

        'user_answer': 'int8', 

        'answered_correctly': 'int8',

        'prior_question_elapsed_time': 'float32', 

        'prior_question_had_explanation': 'boolean'

    }

)



reader
train_df1 = reader.get_chunk(101230332 / 2)

sys.getsizeof(train_df1)
train_df1.memory_usage()
train_df1.drop(['row_id', 'user_id', 'content_id', 'user_answer'], axis=1, inplace=True)

sys.getsizeof(train_df1)
train_df2 = reader.get_chunk(101230332 / 2)

train_df2.drop(['row_id', 'user_id', 'content_id', 'user_answer'], axis=1, inplace=True)
train_df = pd.concat([train_df1, train_df2])

del train_df1, train_df2

gc.collect()

train_df.describe()
train_df_lecture = train_df[train_df['content_type_id'] == 1]

train_df_lecture.shape
train_df_lecture['answered_correctly'].value_counts()
train_df[train_df['answered_correctly'] == -1]['content_type_id'].value_counts()
train_df_lecture['prior_question_elapsed_time'].isnull().count()
train_df_lecture['prior_question_had_explanation'].value_counts()
del train_df_lecture

gc.collect()
print('Number of missing values for every column')

print(train_df.isnull().sum())
train_df['prior_question_had_explanation'].value_counts() / (len(train_df) - 392506)
train_df.groupby('prior_question_had_explanation')['answered_correctly'].mean()
train_df_explanation_null = train_df[train_df['prior_question_had_explanation'].isnull()]

train_df_explanation_null.shape
train_df_explanation_null[train_df_explanation_null['timestamp'] == 0].describe()
train_df_explanation_null[train_df_explanation_null['timestamp'] != 0].describe()
del train_df_explanation_null

gc.collect()
train_df_elapsed_null = train_df[train_df['prior_question_elapsed_time'].isnull()]

train_df_elapsed_null = train_df_elapsed_null[train_df_elapsed_null['content_type_id'] == 0]

train_df_elapsed_null.shape
train_df_timestamp0 = train_df[train_df['timestamp'] == 0]

train_df_timestamp0.shape
train_df_timestamp0['prior_question_had_explanation'].value_counts()
train_df_elapsed_check = train_df[train_df['prior_question_elapsed_time'] < 10000]

train_df_elapsed_check = train_df_elapsed_check.sort_values('prior_question_elapsed_time')

train_df_elapsed_check['prior_question_elapsed_time'].unique()
del train_df_elapsed_null, train_df_timestamp0, train_df_elapsed_check

gc.collect()
train_df['task_container_id'].value_counts()
train_df[train_df['task_container_id'] == 9999]
reader = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv', 

    low_memory=False, 

    chunksize=5295828, 

    dtype={

        'row_id': 'int64', 

        'timestamp': 'int64', 

        'user_id': 'int32', 

        'content_id': 'int16', 

        'content_type_id': 'int8',

        'task_container_id': 'int16', 

        'user_answer': 'int8', 

        'answered_correctly': 'int8',

        'prior_question_elapsed_time': 'float32', 

        'prior_question_had_explanation': 'boolean'

    }

)



reader
train_df_task_container_check = reader.get_chunk(5295828)
train_df_task_container_check[1896755:1896762]
train_df_task_container_check[1929138:1929144]
train_df_task_container_check[3101565:3101572]
train_df_task_container_check[4301504:4301511]
train_df_task_container_check[5295816:5295826]
del train_df_task_container_check

gc.collect()
task_accuracy = train_df.groupby('task_container_id')['answered_correctly'].mean()

task_accuracy = task_accuracy.rolling(50).mean()



fig = px.line(

    task_accuracy, 

    title='Answer correctness by task_container_id', 

    height=600, 

    width=800

)



fig.show()
del task_accuracy, train_df

gc.collect()
reader = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv', 

    low_memory=False, 

    chunksize=101230332 / 2, 

    dtype={

        'row_id': 'int64', 

        'timestamp': 'int64', 

        'user_id': 'int32', 

        'content_id': 'int16', 

        'content_type_id': 'int8',

        'task_container_id': 'int16', 

        'user_answer': 'int8', 

        'answered_correctly': 'int8',

        'prior_question_elapsed_time': 'float32', 

        'prior_question_had_explanation': 'boolean'

    }

)



reader
train_df1 = reader.get_chunk(101230332 / 2)

sys.getsizeof(train_df1)
train_df1.drop(['row_id', 'timestamp', 'task_container_id', 'user_answer', 'prior_question_elapsed_time', 'prior_question_had_explanation'], axis=1, inplace=True)

sys.getsizeof(train_df1)
train_df2 = reader.get_chunk(101230332 / 2)

train_df2.drop(['row_id', 'timestamp', 'task_container_id', 'user_answer', 'prior_question_elapsed_time', 'prior_question_had_explanation'], axis=1, inplace=True)
train_df = pd.concat([train_df1, train_df2])

del train_df1, train_df2

gc.collect()

pd.options.display.float_format = '{:.11g}'.format

train_df.describe()
train_df['user_id'].value_counts()
train_df['content_id'].value_counts()



train_df_questions = train_df[train_df['content_type_id'] == 0]

train_df_content_id_check = train_df_questions['content_id'].value_counts()

train_df_content_id_check = pd.cut(train_df_content_id_check, [0, 1, 10, 100, 1000, 10000, 100000, 210000]).value_counts()

train_df_content_id_check
del train_df_content_id_check

gc.collect()
user_accuracy = train_df_questions.groupby('user_id')['answered_correctly'].mean()



fig = px.histogram(

    user_accuracy, 

    x="answered_correctly",

    nbins=100,

    width=700,

    height=500,

    title='Answer correctness by user'

)



fig.show()
content_accuracy = train_df_questions.groupby('content_id')['answered_correctly'].mean()



fig = px.histogram(

    content_accuracy, 

    x="answered_correctly",

    nbins=100,

    width=700,

    height=500,

    title='Answer correctness by content'

)



fig.show()
del train_df_questions, user_accuracy, train_df

gc.collect()
questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')

questions_df.rename(columns={'question_id': 'content_id'}, inplace='True')

questions_df
questions_df = pd.merge(content_accuracy, questions_df, on='content_id')

questions_df
part_accuracy = questions_df.groupby('part')['answered_correctly'].mean()

part_accuracy.columns = ['part', 'answer_correctness']

part_accuracy
fig = px.bar(

    part_accuracy, 

    title='accuracy by part'

)



fig.show()