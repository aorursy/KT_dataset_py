import os

os.listdir('../input/riiid-test-answer-prediction')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_white"

# the Total number of rows in train.csv are 101M . So lets load a part of it 

train = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", low_memory=False, 

    nrows=5000000,dtype={

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

    })
train.head(10)
train.info()
print(f'There are {train.shape[0]} rows in train data.')

print(f"Total number of Unique users in our train_df is {train['user_id'].nunique()}")
sns.cubehelix_palette(as_cmap=True)



# Lets check how many user saw an explanation and  correct response(s) after answering  previous question bundle.

ax = sns.countplot(x="prior_question_had_explanation", data=train, palette="Set3",hue="prior_question_had_explanation")

ax = sns.countplot(x="answered_correctly", data=train,hue="answered_correctly")

ax = sns.countplot(x="prior_question_had_explanation", data=train, palette="Set3",hue="answered_correctly")

ax = sns.countplot(x="content_type_id", data=train, palette="Set3")

ds = train['user_answer'].value_counts().reset_index()

ds.columns = ['answers', 'percent_of_answers']

ds['percent_of_answers'] /= len(train)

ds = ds.sort_values(['percent_of_answers'])

fig = px.bar(

    ds, 

    x='answers', 

    y='percent_of_answers', 

    orientation='v', 

    title='Percent of user answers', 

    height=500, 

    width=500

)

fig.show()
sns.set()

fig = plt.figure(figsize=(15,6))

fig = sns.kdeplot(train.groupby(by='user_id').count()['row_id'], shade=True, gridsize=50, color='g', legend=False)

fig.figure.suptitle("User_id distribution", fontsize = 20)

plt.xlabel('User_id counts', fontsize=16)

plt.ylabel('Probability', fontsize=16);

fig = px.histogram(

    train, 

    x="prior_question_elapsed_time",

    nbins=100,

    width=700,

    height=600,

    title='Time taken to solve all questions in the previous bundle'

)

fig.show()
ds = train[train['user_id'] == 115]

x = ds['timestamp']

y = ds['prior_question_elapsed_time']

fig,axes = plt.subplots(1,2,figsize=(10,5))

sns.distplot(y,ax=axes[0])

sns.distplot(x,ax=axes[1])
ds.info()
df = train['user_id'].value_counts().reset_index()

df.columns = ['user_id', 'count']

df = df.sort_values(['count'])
df['count'].value_counts()
df = df.sort_values(['count'])

fig = px.bar(

    df.tail(50), 

    x='user_id', 

    y='count', 

    orientation='h', 

    title='Top 40 users', 

    height=800, 

    width=800

)



fig.show()
print(f"Percentage of Lecture actvities = {(train['answered_correctly']==-1).mean()}%")

train_questions_only_df = train[train['answered_correctly']!=-1]

train_questions_only_df['answered_correctly'].mean()

questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
questions.head()
print(f"Total rows in questions metadata dataframe = {questions.shape[0]}")

print(f'Features in Questions dataframe = {questions.shape[1]}')
sns.countplot(x='correct_answer',data=questions)
sns.countplot(x='part',data=questions)
quest_id = questions['question_id'].tolist()

# Need to check the intersection of questions-id and content_id to check whether the user answered correctly/not
lectures = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv")
lectures.head()
sns.countplot(x='type_of',data=lectures)
ds = lectures['tag'].value_counts().reset_index()

ds.columns = ['tag', 'count']

ds['tag'] = ds['tag'].astype(str) + '-'

ds = ds.sort_values(['count'])

fig = px.bar(

    ds.tail(40), 

    x='count', 

    y='tag', 

    orientation='h', 

    title='Top 40 lectures by number of tags', 

    height=800, 

    width=700

)

fig.show()
sns.countplot(x='part',data=lectures,palette='Set3')
ex_sub = pd.read_csv("../input/riiid-test-answer-prediction/example_test.csv")
ex_sub
# sample submission 

sample = pd.read_csv("../input/riiid-test-answer-prediction/example_sample_submission.csv")
sample
grouped_by_user_df = train_questions_only_df.groupby('user_id')

content_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count'] })



user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count'] })
active_users_dict = user_answers_df[user_answers_df[('answered_correctly','count')] >= 20][('answered_correctly','mean')].to_dict()

popular_questions_dict = content_answers_df[content_answers_df[('answered_correctly','count')] >= 20][('answered_correctly','mean')].to_dict()
def predict_sample(user_id, content_id):

    if content_id in popular_questions_dict:

        return popular_questions_dict[content_id]

    if user_id in active_users_dict:

        return active_users_dict[user_id]

    return 0.658

import riiideducation



env = riiideducation.make_env()
iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:

    test_df['answered_correctly'] = test_df.apply(lambda x: predict_sample(x['user_id'], x['content_id']), 1)

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])