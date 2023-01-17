# import libraries
import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# datatable
!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl

#color
from colorama import Fore, Back, Style
y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL

#plotly
#!pip install chart_studio
!pip install ../input/chart-studio/chart_studio-1.0.0-py3-none-any.whl
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()
# List files available
print(f'{y_}{list(os.listdir("../input/riiid-test-answer-prediction"))}{r_}' )
%%time

train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=10**5, 
                       dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',
                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 
                             'prior_question_had_explanation': 'boolean',
                             }
                      )
print(Fore.YELLOW + 'Training data shape: ',Style.RESET_ALL,train_df.shape)
train_df
import gc

del train_df
gc.collect()
%%time

# reading the dataset from raw csv file
import datatable as dt

dt.fread("../input/riiid-test-answer-prediction/train.csv").to_jay("train.jay")

train_df = dt.fread("train.jay").to_pandas()

print(Fore.YELLOW + 'Training data shape: ',Style.RESET_ALL,train_df.shape)
train_df
import riiideducation

# You can only call make_env() once, so don't lose it!
env = riiideducation.make_env()
iter_test = env.iter_test()
iteration = 0
count = 0
for (test_df, sample_prediction_df) in iter_test:
    test_df['answered_correctly'] = 0.5
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
    print(f'{iteration} iteration !!!!')
    iteration += 1
    
    print(len(test_df))
    count += len(test_df)
'''
print(Fore.YELLOW + 'Test data shape: ',Style.RESET_ALL,test_df.shape)

test_df.head()
'''
print(f'{y_}Test data shape: {sr_}{test_df.shape}')

test_df.head()
count
test_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv')
test_df.iloc[[34, 35, 36]]
test_df.iloc[[51, 52, 53]]
test_df.iloc[[67, 68, 69]]
test_df.iloc[[78, 79, 80]]
test_df.iloc[[80, 81, 82]]
question_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')

print(f'{y_}Questions metadata shape: {sr_}{question_df.shape}')
question_df.head()
lectures_df = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')

print(f'{y_}Lectures metadata shape: {sr_}{question_df.shape}')
lectures_df.head()
print(f'{y_}Train Set !!: {sr_}')
print(train_df.info())
print('-------------')
print(f'{y_}Question Set !!: {sr_}')
print(question_df.info())
print('-------------')
print(f'{y_}Lectures Set !!: {sr_}')
print(lectures_df.info())
print(f'Total row_id in Train set: {g_}{train_df["row_id"].count()}{sr_}')
train_df.isnull().sum()
question_df.isnull().sum()
lectures_df.isnull().sum()
print(Fore.YELLOW + "The total user ids are",Style.RESET_ALL,f"{train_df['user_id'].count()},", Fore.BLUE + "from those the unique ids are", Style.RESET_ALL, f"{train_df['user_id'].value_counts().shape[0]}.")
train_df['row_id'].value_counts().max()
train_df['user_id'].value_counts().max()
train_df['content_id'].value_counts().max()
train_df['task_container_id'].value_counts().max()
train_df = train_df[['user_id', 'row_id', 'timestamp', 'content_id', 'content_type_id', 'task_container_id', 'user_answer', 'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation']].drop_duplicates()
train_df.head()
train_df['timestamp'].iplot(kind='hist',
                              xTitle='timestamp', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#FB8072',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the timestamp in the train_df')
train_df.groupby(['user_id'])['timestamp'].max().sort_values(ascending=False)
fig = px.scatter(train_df, x="user_id", y="timestamp", color='user_id')
fig.show()
train_df['content_id'].value_counts()
train_df['content_id'].iplot(kind='hist',
                              xTitle='content_id', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#FB8072',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the content_id column in the Unique Train_df')
train_df.loc[train_df['content_id'] == 4120, 'user_answer'].value_counts()
question_df.loc[question_df['question_id'] == 4120]
fig = px.scatter(train_df, x="user_id", y="content_id", color='content_type_id')
fig.show()
train_df['content_type_id'].value_counts()
train_df['content_type_id'].value_counts().iplot(kind='bar',
                                          yTitle='Count', 
                                          linecolor='black', 
                                          opacity=0.7,
                                          color='blue',
                                          theme='pearl',
                                          bargap=0.8,
                                          gridcolor='white',
                                          title='Distribution of the Content_type_id column in Train_df')
# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=train_df['content_type_id'].value_counts().index, values=train_df['content_type_id'].value_counts(), pull=[0, 0.2])])
fig.show()
fig = px.scatter(train_df, x="content_id", y="user_id", color='user_id')
fig.show()
train_df['task_container_id']
train_df['task_container_id'].iplot(kind='hist',
                              xTitle='task_container_id', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#FB8072',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the task_container_id in the train_df')
fig = px.scatter(train_df, x="task_container_id", y="prior_question_elapsed_time", color='user_id')
fig.show()
train_df['task_container_id'].value_counts()
train_df.loc[train_df['task_container_id'] == 15, 'user_answer'].value_counts()
question_df.loc[question_df['question_id'] == 15]
train_df.loc[train_df['task_container_id'] == 5283, 'user_answer'].value_counts()
question_df.loc[question_df['question_id'] == 5283]
train_df['user_answer'].value_counts()
train_df['user_answer'].value_counts().iplot(kind='bar',
                                          yTitle='Count', 
                                          linecolor='black', 
                                          opacity=0.7,
                                          color='red',
                                          theme='pearl',
                                          bargap=0.8,
                                          gridcolor='white',
                                          title='Distribution of the user_answer column in Train_df')
ds = train_df['user_answer'].value_counts().reset_index()
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
fig = px.scatter(train_df, x="user_answer", y="content_type_id", color='user_id')
fig.show()
train_df['answered_correctly'].value_counts()
train_df['answered_correctly'].value_counts().iplot(kind='bar',
                                          yTitle='Count', 
                                          linecolor='black', 
                                          opacity=0.7,
                                          color='blue',
                                          theme='pearl',
                                          bargap=0.8,
                                          gridcolor='white',
                                        title='Distribution of the answered_correctly column in Train_df')
plt.figure(figsize = (16,12))

a = sns.countplot(data=train_df, x='answered_correctly', hue='prior_question_had_explanation')


for p in a.patches:
    a.annotate(format(p.get_height(), ','), 
           (p.get_x() + p.get_width() / 2., 
            p.get_height()), ha = 'center', va = 'center', 
           xytext = (0, 4), textcoords = 'offset points')

plt.title('Answers result with and without explanations', fontsize=20)
plt.xlabel('Answered_correctly', fontsize = 16)
sns.despine(left=True, bottom=True);
plt.figure(figsize=(16,8))
sns.countplot(train_df['user_answer'], hue=train_df['answered_correctly'],palette='Set1',**{'hatch':'-','linewidth':0.5})
plt.title('User_Answer vs Correctness', fontsize = 20)
plt.show()
fig = px.scatter(train_df, x="answered_correctly", y="task_container_id", color='user_id')
fig.show()
train_df['prior_question_elapsed_time']
train_df['prior_question_elapsed_time'].iplot(kind='hist',
                              xTitle='prior_question_elapsed_time', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#FB8072',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the prior_question_elapsed_time column in the Unique Train_df')
fig = px.scatter(train_df, x="content_id", y="prior_question_elapsed_time", color='user_id')
fig.show()
train_df['prior_question_had_explanation'].value_counts()
train_df['prior_question_had_explanation'].value_counts().iplot(kind='bar',
                                          yTitle='Count', 
                                          linecolor='black', 
                                          opacity=0.7,
                                          color='red',
                                          theme='pearl',
                                          bargap=0.8,
                                          gridcolor='white',
                                          title='Distribution of the prior_question_had_explanation column in Train_df')
temp_train = train_df.groupby('user_id').agg({'answered_correctly': 'sum', 'row_id':'count'})
plt.figure(figsize = (16,8))
sns.distplot((temp_train.answered_correctly * 100)/temp_train.row_id)
plt.title('Distribution of correct answers percentage by each user', fontdict = {'size': 16})
plt.xlabel('Percentage of correct answers', size = 12)
corrmat = train_df.corr() 
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap = 'RdYlBu_r', linewidths = 0.5) 
question_df.head()
question_df['bundle_id'].iplot(kind='hist',
                              xTitle='bundle_id', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#FB8072',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the bundle_id in the question_df')
question_df['correct_answer'].iplot(kind='hist',
                              xTitle='correct_answer', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#098060',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the correct_answer in the question_df')
fig = px.scatter(question_df, x="bundle_id", y="correct_answer", color='question_id')
fig.show()
question_df['part'].iplot(kind='hist',
                              xTitle='correct_answer', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#FB8072',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the part in the question_df')
fig = px.scatter(question_df, x="correct_answer", y="part", color='part')
fig.show()
corrmat = question_df.corr() 
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap = 'RdYlBu_r', linewidths = 0.5) 
lectures_df.head()
lectures_df['tag'].iplot(kind='hist',
                              xTitle='tag', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#FB8072',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the tag in the lectures_df')
lectures_df['part'].iplot(kind='hist',
                              xTitle='part', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#098060',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the part in the lectures_df')
lectures_df['type_of'].iplot(kind='hist',
                              xTitle='part', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#FB8072',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the type_of in the lectures_df')
fig = px.scatter(lectures_df, x="type_of", y="part", color='lecture_id')
fig.show()
fig = px.bar(lectures_df, x='type_of', color=lectures_df['type_of'], labels={'value':'type_of'}, title='Type of lectures distribution Overall')
fig.show()
fig = px.bar(lectures_df, x='type_of', color=lectures_df['type_of'], labels={'value':'type_of'}, title='Type of lectures distribution based on each part', facet_col='part')
fig.show()
corrmat = lectures_df.corr() 
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap = 'RdYlBu_r', linewidths = 0.5) 
import pandas_profiling as pdp
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=10**5, 
                       dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',
                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 
                             'prior_question_had_explanation': 'boolean',
                             }
                      )

test_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv')
profile_train_df = pdp.ProfileReport(train_df)
profile_train_df
profile_test_df = pdp.ProfileReport(test_df)
profile_test_df
profile_question_df = pdp.ProfileReport(question_df)
profile_question_df
profile_lectures_df = pdp.ProfileReport(lectures_df)
profile_lectures_df
content_acc = train_df.query('answered_correctly != -1').groupby('content_id')['answered_correctly'].mean().to_dict()
def add_content_acc(x):
    if x in content_acc.keys():
        return content_acc[x]
    else:
        return 0.5


for (test_df, sample_prediction_df) in iter_test:
    test_df['answered_correctly'] = test_df['content_id'].apply(add_content_acc).values
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])