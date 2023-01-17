# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import plotly.express as px

from collections import Counter as count



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense

from sklearn.model_selection import KFold





debug = False


if debug:

    read_num = 10**6

else:

    read_num = 10**7



train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=read_num, 

                       dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

                             'prior_question_had_explanation': 'boolean',

                             }

                      )

test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')

submit = pd.read_csv('../input/riiid-test-answer-prediction/example_sample_submission.csv')



print('Train shapes: ', train.shape)

print('Test shapes: ', test.shape)
train.head(10)
test.head(10)
submit
fig = px.histogram(

    train, 

    "task_container_id", 

    nbins=25, 

    title='task_container_id column distribution', 

    width=700,

    height=500

)

fig.show()



fig = px.histogram(

    train, 

    "timestamp", 

    nbins=25, 

    title='timestamp column distribution', 

    width=700,

    height=500

)

fig.show()
fig = px.histogram(

    train, 

    "prior_question_elapsed_time", 

    nbins=25, 

    title='prior_question_elapsed_time column distribution', 

    width=700,

    height=500

)

fig.show()



ds = train['answered_correctly'].value_counts().reset_index()

ds.columns = ['answered_correctly', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names="answered_correctly", 

    title='answered_correctly bar chart', 

    width=500, 

    height=500

)

fig.show()
ds = train['prior_question_had_explanation'].value_counts().reset_index()

ds.columns = ['prior_question_had_explanation', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names="prior_question_had_explanation", 

    title='prior_question_had_explanation bar chart', 

    width=500, 

    height=500

)

fig.show()


ds = train['user_answer'].value_counts().reset_index()

ds.columns = ['user_answer', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names="user_answer", 

    title='user_answer bar chart', 

    width=500, 

    height=500

)

fig.show()


ds = train['content_type_id'].value_counts().reset_index()

ds.columns = ['content_type_id', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names="content_type_id", 

    title='content_type_id bar chart', 

    width=500, 

    height=500

)

fig.show()