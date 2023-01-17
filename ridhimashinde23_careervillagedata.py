# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
answer_scores = pd.read_csv('../input/answer_scores.csv')

answers = pd.read_csv('../input/answers.csv')

comments = pd.read_csv('../input/comments.csv')
answer_scores.head(3)
answer_scores.describe()
answer_scores.max()
answer_scores.info()
score_count = answer_scores['score'].value_counts().sort_index()

score_count
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



score_count.plot.hist(bins = 10)
answers.head(3)
answers.info()
answers['answers_author_id'].unique()
len(answers['answers_author_id'].unique())
longest_ans = answers['answers_body'].str.len().max()

shortest_ans = answers['answers_body'].str.len().min()



print('The longest answer has ',longest_ans, 'characters')

print('The shortest answer has',shortest_ans, 'characters')
oldest_ans = answers['answers_date_added'].min()

newest_ans = answers['answers_date_added'].max()



print('The oldest answer is',oldest_ans)

print('The newest answer is',newest_ans)
ans_author_grouped = answers.groupby('answers_author_id')

ans_author_grouped.size()
print('Maximum number of answers by a single author:',ans_author_grouped.size().max())
ans_date = pd.to_datetime(answers['answers_date_added'])

ans_date.head()
comments.head(3)
comments.shape
longest_cmt = comments['comments_body'].str.len().max()

shortest_cmt = comments['comments_body'].str.len().min()



print('The longest comment has ',longest_cmt, 'characters')

print('The shortest comment has',shortest_cmt, 'characters')
newest_cmt = comments['comments_date_added'].max()

oldest_cmt = comments['comments_date_added'].min()



print('The longest comment has ',newest_cmt, 'characters')

print('The shortest comment has',oldest_cmt, 'characters')
emails = pd.read_csv('../input/emails.csv')

group_memberships = pd.read_csv('../input/group_memberships.csv')

groups = pd.read_csv('../input/groups.csv')
emails.head(3)
emails.shape
group_memberships.head(3)
group_memberships.info()
groups.head(3)
groups.shape
groups['groups_group_type'].unique()
grouped = groups.groupby('groups_group_type').count()

grouped
matches = pd.read_csv('../input/matches.csv')

professionals = pd.read_csv('../input/professionals.csv')

question_scores = pd.read_csv('../input/question_scores.csv')
matches.head(3)

professionals.head(3)

professionals.info()
professionals.isnull().any()
#Number of missing records in professionals.csv



professionals.isnull().sum()
prof_loc = professionals['professionals_location'].isnull().sum()

print(prof_loc)
prof_id = len(professionals['professionals_id'])

print(prof_id)
def percentage(int1, int2):

    div = (int1/int2) * 100

    return round(div, 2)
percentage(prof_loc, prof_id)
professionals.shape
professionals.dropna()
professionals['professionals_location'].value_counts()

professionals['professionals_date_joined'] = pd.to_datetime(professionals['professionals_date_joined'])

professionals_copy = professionals.set_index(keys='professionals_date_joined')
professionals_yearly_trend = professionals_copy['professionals_id'].groupby([professionals_copy.index.year]).count()

professionals_monthly_trend = professionals_copy['professionals_id'].groupby([professionals_copy.index.month]).count()



professionals_copy['month_year'] = professionals_copy.index.to_period('M').astype('str')

professionals_monthly_distibution = professionals_copy[['month_year', 'professionals_id']].groupby(by='month_year').count()



x = professionals_monthly_distibution.index.values
fig,ax = plt.subplots(figsize=(16, 12))

sns.barplot(professionals_yearly_trend.index.values, 

            professionals_yearly_trend.values, 

            ax=ax)



ax.set_title('Professionals joining by the year')

ax.set_ylabel('Number')

ax.set_xlabel('Year')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=15);

#ax.axhline(0, color="k", clip_on=False)
fig,ax = plt.subplots(figsize=(16, 12))

sns.barplot(professionals_monthly_trend.index.values, 

            professionals_monthly_trend.values, 

            ax=ax)



ax.set_title('Professionals joining by the month')

ax.set_ylabel('Number')

ax.set_xlabel('Months')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=15);

#ax.axhline(0, color="k", clip_on=False)
prof_locations = professionals['professionals_location'].value_counts().head(10)

print(prof_locations)

professionals['professionals_location'] = professionals[~pd.isnull(professionals['professionals_location'])]['professionals_location'].astype(str)

professionals_location = professionals['professionals_location'].value_counts().sort_values(ascending=False)
import seaborn as sns



fig, ax = plt.subplots(figsize=(25, 9))

sns.barplot(professionals_location.index.values[:10], 

            professionals_location.values[:10], 

            ax=ax)



ax.set_title('Professionals by location')

ax.set_ylabel('Count')

ax.set_xlabel('Location')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=15);

ax.axhline(0, color="k", clip_on=False)
professionals['professionals_industry'].unique()
#prof_grouped_industry = professionals.groupby('professionals_industry')

#prof_grouped_industry.size()



prof_industry = professionals['professionals_industry'].value_counts().head(10)

print(prof_industry)
professionals['professionals_industry'] = professionals[~pd.isnull(professionals['professionals_industry'])]['professionals_industry'].astype(str)

professionals_industry = professionals['professionals_industry'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(25, 9))

sns.barplot(professionals_industry.index.values[:10], 

            professionals_industry.values[:10], 

            ax=ax)



ax.set_title('Professionals by Industry')

ax.set_ylabel('Count')

ax.set_xlabel('Industry')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=15);

ax.axhline(0, color="k", clip_on=False)
professionals['professionals_headline'].unique()
professionals['professionals_headline'] = professionals[~pd.isnull(professionals['professionals_headline'])]['professionals_headline'].astype(str)

professionals_headline = professionals['professionals_headline'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(25, 9))

sns.barplot(professionals_headline.index.values[:10], 

            professionals_headline.values[:10], 

            ax=ax)



ax.set_title('Professionals by headline')

ax.set_ylabel('Count')

ax.set_xlabel('Headline')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=15);

ax.axhline(0, color="k", clip_on=False)
question_scores.head(3)
question_scores['score'].value_counts().sort_index()
questions = pd.read_csv('../input/questions.csv')

school_memberships = pd.read_csv('../input/school_memberships.csv')

students = pd.read_csv('../input/students.csv')

tag_questions = pd.read_csv('../input/tag_questions.csv')

tag_users = pd.read_csv('../input/tag_users.csv')

tags = pd.read_csv('../input/tags.csv')
questions.head(3)
questions.shape
longest_ques = questions['questions_body'].str.len().max()

shortest_ques = questions['questions_body'].str.len().min()



print('The longest question has ',longest_ques, 'characters')

print('The shortest question has',shortest_ques, 'characters')
newest_ques = questions['questions_date_added'].max()

oldest_ques = questions['questions_date_added'].min()



print('The newest question was received at',newest_ques)

print('The oldest question was received at',oldest_ques)
school_memberships.head(3)
school_memberships.shape
students.head(3)
students.shape
students['students_location'].unique()
students_grouped_location = students.groupby('students_location')

students_grouped_location.size()
students['students_location'] = students[~pd.isnull(students['students_location'])]['students_location'].astype(str)

students_location = students['students_location'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(25, 9))

sns.barplot(students_location.index.values[:10], 

            students_location.values[:10], 

            ax=ax)



ax.set_title('Student location')

ax.set_ylabel('Count')

ax.set_xlabel('Locations')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=15);

ax.axhline(0, color="k", clip_on=False)
students['students_date_joined'] = pd.to_datetime(students['students_date_joined'])

students_copy = students.set_index(keys='students_date_joined')

students_yearly_trend = students_copy['students_id'].groupby([students_copy.index.year]).count()

students_monthly_trend = students_copy['students_id'].groupby([students_copy.index.month]).count()



students_copy['month_year'] = students_copy.index.to_period('M').astype('str')

students_monthly_distibution = students_copy[['month_year', 'students_id']].groupby(by='month_year').count()



x = students_monthly_distibution.index.values



fig,ax = plt.subplots(figsize=(16, 12))

sns.barplot(students_yearly_trend.index.values, 

            students_yearly_trend.values, 

            ax=ax)



ax.set_title('Students joining by the year')

ax.set_ylabel('Number')

ax.set_xlabel('Year')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=15);

#ax.axhline(0, color="k", clip_on=False)

fig,ax = plt.subplots(figsize=(16, 12))

sns.barplot(students_monthly_trend.index.values, 

            students_monthly_trend.values, 

            ax=ax)



ax.set_title('Students joining by the month')

ax.set_ylabel('Number')

ax.set_xlabel('Months')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=15);

#ax.axhline(0, color="k", clip_on=False)

tag_questions.head(3)
tag_questions.shape
tag_users.head(3)
tag_users.shape
tags.head(3)
tags.shape
tags['tags_tag_name'].unique()
tags_count = tags['tags_tag_name'].value_counts().head(10)

print(tags_count)
tags['tags_tag_name'] = tags[~pd.isnull(tags['tags_tag_name'])]['tags_tag_name'].astype(str)

tags_tag_name = tags['tags_tag_name'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(25, 9))

sns.barplot(tags_tag_name.index.values[:10], 

            tags_tag_name.values[:10], 

            ax=ax)



ax.set_title('Top tag names')

ax.set_ylabel('Count')

ax.set_xlabel('Tag names')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=15);

ax.axhline(0, color="k", clip_on=False)