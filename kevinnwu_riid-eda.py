import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline
lectures = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')

ex_submission = pd.read_csv('../input/riiid-test-answer-prediction/example_sample_submission.csv')

ex_test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')

questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')

train = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', nrows = 99999)
train.head()
train = train.drop(train[train['content_type_id']==1].index)
train
train = train.rename(columns={'content_id':'question_id'})
train
train = train.merge(questions)
train
fig = px.histogram(train,

                   x='user_id', 

                   histfunc='count',

                  )

fig.update_layout(showlegend =False,

                 width=1000,

                 height=600)

fig.show()
fig = px.histogram(train,

                   x='user_id', 

                   y='timestamp',

                   histfunc='max',

                  )

fig.update_layout(showlegend =False,

                 width=1000,

                 height=600)

fig.show()
fig = px.histogram(train,

                   x='user_id', 

                   y='answered_correctly',

                   histfunc='avg',

                  )

fig.update_layout(showlegend =False,

                 width=1000,

                 height=600)

fig.show()
fig = px.histogram(train,

                   x='question_id', 

                   y='answered_correctly',

                   histfunc='avg',

                  )

fig.update_layout(showlegend =False,

                 width=1000,

                 height=600)

fig.show()
fig = px.histogram(train,

                   x='part', 

                   y='answered_correctly',

                   histfunc='avg',

                  )

fig.update_layout(showlegend =False,

                 width=1000,

                 height=600)

fig.show()
plt.figure(figsize=(20,10))

sns.barplot(x='index', y ='answered_correctly', data=train.corr().drop('answered_correctly').reset_index())
train = train.drop(['row_id', 'content_type_id', 'user_answer', 'correct_answer'], axis=1)
question_answered_correctly = train.groupby('question_id').mean()['answered_correctly'].reset_index()

question_answered_correctly
question_answered_correctly = question_answered_correctly.rename(columns={'answered_correctly':'PQAC'})

train = train.merge(question_answered_correctly)
part_answered_correctly = train.groupby('part').mean()['answered_correctly'].reset_index()

part_answered_correctly
part_answered_correctly = part_answered_correctly.rename(columns={'answered_correctly':'PPQAC'})

train = train.merge(part_answered_correctly)
user_answered_correctly = train.groupby('user_id').mean()['answered_correctly'].reset_index()

user_answered_correctly
user_answered_correctly = user_answered_correctly.rename(columns={'answered_correctly':'UAC'})

train = train.merge(user_answered_correctly)
avg_user_answered = (train.groupby('user_id').max()['timestamp']/train.groupby('user_id').count()['timestamp']).reset_index()

avg_user_answered
avg_user_answered = avg_user_answered.rename(columns={'timestamp':'AUA'})

train = train.merge(avg_user_answered)
train['AUAC'] = train['UAC']*train['AUA']
plt.figure(figsize=(20,10))

sns.barplot(x='index', y ='answered_correctly', data=train.corr().drop('answered_correctly').reset_index())
from datetime import datetime
train['datetime'] = pd.to_datetime(train['timestamp'], unit='ms')
train.index = train['datetime']
z = train.groupby('user_id').resample('1D').count()[0:20]['answered_correctly'].reset_index()
z = z.rename(columns={'answered_correctly':'question/day', 'datetime': 'date_id'})

z
train = train.reset_index(drop=True)
train['date_id'] = train['datetime'].dt.date
train[20:40]
sns.scatterplot(x='timestamp', y='answered_correctly',data=train)
fig = px.line(train,

              x='timestamp', 

              y='answered_correctly',

              color='user_id',

              #text='content_id',

                 #symbol='Outlet_Type',

                 #text='Outlet_Identifier',

             )#.update_yaxes(categoryorder='total ascending')



fig.update_traces(marker=dict(size=12,),

                  textposition='top center',

                  textfont=dict(family='Arial',size=12),

              

                 )

fig.update_layout(

    height=600,

)



fig.show()