# Load in libraries



import warnings

warnings.filterwarnings('ignore')

import os



#libraries for handling data

import pandas as pd

import numpy as np

import math

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler()

from sklearn.preprocessing import RobustScaler

rscaler = RobustScaler()

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

import datetime

from datetime import datetime, time



#libraries for data visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.pylab as pylab

import seaborn as sns



#libaries for modelling

# Regression Modelling Algorithms

import statsmodels.api as sm

#from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
groups = pd.read_csv('../input/groups.csv')

group_memberships = pd.read_csv('../input/group_memberships.csv')

school_memberships = pd.read_csv('../input/school_memberships.csv')

tags = pd.read_csv('../input/tags.csv')

answers = pd.read_csv('../input/answers.csv')

emails = pd.read_csv('../input/emails.csv')

comments = pd.read_csv('../input/comments.csv')

questions = pd.read_csv('../input/questions.csv')

matches = pd.read_csv('../input/matches.csv')

professionals = pd.read_csv('../input/professionals.csv')

students = pd.read_csv('../input/students.csv')

tag_questions = pd.read_csv('../input/tag_questions.csv')

tag_users =pd.read_csv('../input/tag_users.csv')
def get_meta_info_about_columns_and_tables(df_arr, df_name_arr):

    tables = []

    columns = []

    for df, name in zip(df_arr, df_name_arr):

        columns.extend(df.columns.values)

        tables.extend([name] * len(df.columns))

    return pd.DataFrame({'table': tables, 'column': columns})
tables_columns_info = get_meta_info_about_columns_and_tables(

    [

        professionals,

        tag_users,

        students,

        tag_questions,

        groups,

        emails,

        group_memberships,

        answers,

        comments,

        matches,

        tags,

        questions,

        school_memberships

    ],

    [

        'professionals',

        'tag_users',

        'students',

        'tag_questions',

        'groups',

        'emails',

        'group_memberships',

        'answers',

        'comments',

        'matches',

        'tags',

        'questions',

        'school_memberships'

    ]

)
tables_columns_info.head(10)
tables_columns_info[tables_columns_info.column.str.contains('group')]
professionals['professionals_date_joined'] = pd.to_datetime(professionals['professionals_date_joined'], errors='coerce')

students['students_date_joined'] = pd.to_datetime(students['students_date_joined'], errors='coerce')

emails.emails_date_sent = pd.to_datetime(emails.emails_date_sent, errors='coerce')

answers.answers_date_added = pd.to_datetime(answers.answers_date_added , errors='coerce')

comments.comments_date_added = pd.to_datetime(comments.comments_date_added, errors='coerce')

questions.questions_date_added = pd.to_datetime(questions.questions_date_added, errors = 'coerce')
groups.head(3)
groups.info()
groups.isnull().sum()
answers.head()
students.head(3)
students.info()
students.isnull().sum()
students.students_location.nunique()
students_locations_top = students.students_location.value_counts().sort_values(ascending = False).head(10)
students_locations_top
students_locations_top.plot.bar()
groups.columns
groups.groups_group_type.unique()
groups.groups_group_type.nunique()
groups.groups_group_type.value_counts()
#groups.groups_group_type.value_counts().sort_values(ascending=True).plot.pie(title='Group Types')

groups.groups_group_type.value_counts().sort_values(ascending=True).plot(kind='pie',autopct='%1.1f%%', title='Group Types')

plt.xlabel('')

plt.ylabel('')
#groups.groups_group_type.value_counts().sort_values(ascending=True).plot.barh(title='Group Types')

groups.groups_group_type.value_counts().sort_values(ascending=True).plot(kind='barh', title='horizoneal bar graph')

plt.xlabel('counts')

plt.ylabel('group types')
#groups.groups_group_type.value_counts().plot.bar()

#groups.groups_group_type.value_counts().sort_values(ascending=True).plot.bar(title='group types')

groups.groups_group_type.value_counts().sort_values(ascending=True).plot(kind='bar', title='group types', figsize=(5, 5))
temp = groups.groups_group_type.value_counts()

temp = temp.reset_index()
temp.head()
sns.barplot(x="groups_group_type", y='index', data=temp, color="cyan")
emails.head()
emails.columns
emails.emails_frequency_level.unique()
professionals.head()
professionals.professionals_location.nunique()
professionals_location_top = professionals.professionals_location.value_counts().sort_values(ascending=True).head(10)

professionals_location_top
professionals_location_topChart = professionals_location_top.plot.pie(autopct='%1.0f%%')

plt.xlabel('')

plt.ylabel('')
students.head()
students.students_id.count()
students.students_date_joined.isnull().sum()
students['students_date_joined_year'] = students['students_date_joined'].dt.year

students['students_date_joined_month'] = students.students_date_joined.dt.month

students['students_date_joined_day'] = students.students_date_joined.dt.day
students_count =students['students_id'].groupby(students['students_date_joined_year']).count()

students_count
students_count.plot.line()
school_memberships.head()
school_memberships.school_memberships_school_id.nunique()
questions.head()
questions.quest_added_year = questions.questions_date_added.dt.year
questions.questions_id.groupby(questions.quest_added_year).count().plot(kind='bar')
np.cumsum(questions.questions_id.groupby(questions.quest_added_year).count()).plot(kind='bar')
df = pd.merge(questions, answers, how='left', left_on='questions_id', right_on='answers_question_id')

df = pd.merge(df, tag_questions, how ='left', left_on='questions_id', right_on='tag_questions_question_id')

df = pd.merge(df, tags, how='left', left_on='tag_questions_tag_id', right_on='tags_tag_id')

df = pd.merge(df, group_memberships, how='left', left_on='answers_author_id', right_on='group_memberships_user_id')

df = pd.merge(df, groups,how='left', left_on='group_memberships_group_id', right_on='groups_id')
df.head().transpose()
df.questions_date_added = pd.to_datetime(df.questions_date_added, errors='coerce')

df.quest_added_date = df.questions_date_added.dt.date
g1 =df.questions_id.groupby(df.quest_added_date).count()

g1.plot()
df.tags_tag_name.nunique()
top_tags = df.questions_id.groupby(df.tags_tag_name).count().sort_values(ascending=False).head(10)
top_tags.plot.bar()
df['duration_answers'] = (pd.to_datetime(df.answers_date_added) - pd.to_datetime(df.questions_date_added) ).dt.days
df.duration_answers.describe()
df.duration_answers.groupby(df.groups_group_type).describe()
sns.distplot(df.duration_answers.dropna())
sns.kdeplot(df.duration_answers.dropna(), shade=True, color='r')
sns.kdeplot(df.duration_answers.dropna()[df.tags_tag_name=='college'], label='college', shade=False)

sns.kdeplot(df.duration_answers.dropna()[df.tags_tag_name=='career'], label='career', shade=False)

sns.kdeplot(df.duration_answers.dropna()[df.tags_tag_name=='engineering'], label='engineering', shade=False)
sns.kdeplot(df.duration_answers.dropna()[df.groups_group_type=='competition'], label='competition')

#sns.kdeplot(df.duration_answers.dropna()[df.groups_group_type=='youth program'], label='youth program')

sns.kdeplot(df.duration_answers.dropna()[df.groups_group_type=='mentorship program'], label='mentorship program')
df.duration_answers.groupby(df.groups_group_type).mean()