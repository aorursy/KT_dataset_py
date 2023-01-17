import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

#Plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



%matplotlib inline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
students = pd.read_csv('../input/students.csv')

profs = pd.read_csv('../input/professionals.csv')

groups = pd.read_csv('../input/groups.csv')

comments = pd.read_csv('../input/comments.csv')

school_mem = pd.read_csv('../input/school_memberships.csv')

tags = pd.read_csv('../input/tags.csv')

emails = pd.read_csv('../input/emails.csv', parse_dates = ['emails_date_sent'])

group_mem = pd.read_csv('../input/group_memberships.csv')

answers = pd.read_csv('../input/answers.csv')

matches = pd.read_csv('../input/matches.csv')

questions = pd.read_csv('../input/questions.csv')

tag_users = pd.read_csv('../input/tag_users.csv')

tag_ques = pd.read_csv('../input/tag_questions.csv')
QnA = questions.merge(answers, left_on='questions_id', right_on='answers_question_id')

QnA_prof = QnA.merge(profs, left_on='answers_author_id', right_on='professionals_id')

QnA_prof.head()
QnA_prof.groupby('professionals_id')['professionals_industry'].unique().value_counts()[:30].plot(kind='bar', figsize=(15,8));

plt.title('Industry the Volunteers belong to', fontsize=18);
QnA_prof.groupby('professionals_id')['professionals_location'].unique().value_counts()[:30].plot(kind='bar', figsize=(15,8));

plt.title('Location of Volunteers', fontsize=18);
QnA_prof.groupby('professionals_id')['professionals_headline'].unique().value_counts()[:30].plot(kind='bar', figsize=(15,8));

plt.title('Profession of Volunteers', fontsize=18);
emails.head()
def extract_date(df, column):

    '''

    df takes a dataframe and column takes a string.

    '''

    df['year'] = df[column].apply(lambda x: x.year)

    df['month'] = df[column].apply(lambda x: x.month)

    df['day'] = df[column].apply(lambda x: x.day)
extract_date(emails, 'emails_date_sent')
emails.head()
emails.groupby('emails_recipient_id')['month'].count().plot(kind='bar',figsize=(15,8));