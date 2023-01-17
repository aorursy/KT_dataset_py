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
import seaborn as sns 

import matplotlib.pyplot as plt

import plotly.express as px
train=pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',dtype={'row_id':'int64',\

                                                                               'timestamp':'int64',\

                                                                               'user_id':'int32',\

                                                                               'content_id':'int16',\

                                                                               'content_type_id':'int8',\

                                                                               'task_container_id':'int16',\

                                                                               'user_answer':'int8',\

                                                                               'answered_correctly':'int8',\

                                                                               'prior_question_elapsed_time':'float32',\

                                                                               'prior_question_had_explanation':'boolean'})
train.head()
train.sort_values(['timestamp'],ascending=True)
table=pd.DataFrame({col:train[col].isna().sum() for col in train.columns},index=["missing values counts"])

table
#We know for users first question bundle, prior_question_elapsed_time should be null and prior_question_had_explanation should be False.

train['prior_question_elapsed_time'].fillna(0,inplace=True)

train['prior_question_had_explanation'].fillna(False,inplace=True)
table=pd.DataFrame({col:train[col].isna().sum() for col in train.columns},index=["missing values counts"])

table
# Number of students on this dataset.

users=len(train['user_id'].unique())

print('There is data of {} students in the training set '.format(users))
# percentil of the question and lecture events in the training dataset.

events={0:"question",1:"lecture"}

labels=list(events.values())

val=[train['content_type_id'].value_counts().loc[0],train['content_type_id'].value_counts().loc[1]]
fig=px.pie(names=labels,values=val,title="events percentile")

fig.update_layout(title={'x':0.475,'y':0.9,'xanchor':'center','yanchor':'top'})

fig.show()
# substructe the data concern onlu the question events.

train=train[train['content_type_id']==0]
tr_u=train.groupby(['user_id']).agg({'answered_correctly':'mean'}).sort_values(['answered_correctly'],\

                                                                              ascending=False)
fig=px.histogram(tr_u,x='answered_correctly',title='distribution of performance  level of students ',\

                 nbins=10,histnorm='probability')

fig.update_layout(title={'x':0.475,'y':0.9,'xanchor':'center','yanchor':'top'})

fig.show()