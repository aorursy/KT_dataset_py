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
professionals = pd.read_csv('../input/professionals.csv')
professionals.head()
professionals.describe()
professionals.groupby('professionals_location')['professionals_id'].count().reset_index()
professionals.groupby('professionals_headline')['professionals_id'].count().reset_index()
professionals.groupby('professionals_industry')['professionals_id'].count().sort_values(ascending=False).head(20).reset_index()
professionals.groupby('professionals_industry')['professionals_id'].count().sort_values(ascending=False).tail(20).reset_index()
students = pd.read_csv('../input/students.csv')
students.head()
students.describe()
students.groupby('students_location')['students_id'].count().sort_values(ascending=False).head(20).reset_index()
students.groupby('students_location')['students_id'].count().sort_values(ascending=False).tail(30).reset_index()
### Above are the places where there's still less question
groups = pd.read_csv('../input/groups.csv')
groups.head()
groups.groupby('groups_group_type').count().reset_index()
tag_users = pd.read_csv('../input/tag_users.csv')
tag_users.head()
tag_users.describe()
tag_users.shape
tag_users.groupby('tag_users_tag_id')['tag_users_tag_id'].count().sort_values(ascending=False).head(20)
tag_questions = pd.read_csv('../input/tag_questions.csv')
tag_questions.head()
tag_questions.shape
emails = pd.read_csv('../input/emails.csv')
emails.head()
emails.groupby('emails_frequency_level')['emails_id'].count().reset_index()
emails.groupby('emails_recipient_id')['emails_id'].count().sort_values(ascending=False).head(10)
emails['emails_id'].nunique()
emails.shape
answers = pd.read_csv('../input/answers.csv')
answers.head()
answers.shape
questions = pd.read_csv('../input/questions.csv')
questions.shape
questions.head()
questions.shape
comments = pd.read_csv('../input/comments.csv')
comments.shape
comments.head()
comments['comments_body'][4]
comments['comments_body'][2]
comments['comments_body'][121]
matches = pd.read_csv('../input/matches.csv')
matches.head()
matches.describe()
matches['matches_email_id'].nunique()
matches['matches_question_id'].nunique()
tags = pd.read_csv('../input/tags.csv')
tags.head()
tags['tags_tag_name'].nunique()
tags.shape
import re

[str(tag) for tag in tags['tags_tag_name'].tolist() if re.search(r'computer', str(tag))][:20]
school_membership = pd.read_csv('../input/school_memberships.csv')
group_membership = pd.read_csv('../input/group_memberships.csv')
school_membership.head()
group_membership.head()
school_membership['school_memberships_school_id'].nunique()
school_membership.groupby('school_memberships_school_id').count().reset_index().head(10)
questions.head()
answers.head()
answers = answers.rename(columns={'answers_question_id': 'questions_id'})
questions.shape, answers.shape
merge_qna = pd.merge(questions, answers, on='questions_id', how='left')
merge_qna = merge_qna[['questions_id', 'questions_author_id', 

                       'answers_author_id', 'questions_title', 

                       'questions_body', 'answers_body', 

                       'questions_date_added', 'answers_date_added']]
merge_qna.shape
merge_qna.head()
merge_qna_professionals = pd.merge(merge_qna, professionals, 

                                   left_on='answers_author_id', 

                                   right_on='professionals_id', 

                                   how='left')
merge_qna_professionals = merge_qna_professionals[['questions_id', 'questions_author_id', 

                                                   'answers_author_id', 'questions_title', 

                                                   'questions_body', 'answers_body', 

                                                   'questions_date_added', 'answers_date_added',

                                                   'professionals_location', 'professionals_industry']]
merge_qna_professionals.head()
merge_qna_professionals_students = pd.merge(merge_qna_professionals, students, 

                                             left_on='questions_author_id', 

                                             right_on='students_id', 

                                             how='left')
merge_qna_professionals_students.head()
row1 = merge_qna_professionals_students.loc[1, ['questions_body', 'answers_body', 

                                                'students_id', 'students_location']]
row2 = merge_qna_professionals_students.loc[2, ['questions_body', 'answers_body', 

                                                'students_id', 'students_location']]
row1['questions_body']
row1['answers_body']
row1['students_location']
row2['questions_body']
row2['answers_body']
row2['students_location']
merge_qna_professionals_students = merge_qna_professionals_students[['questions_id', 'questions_author_id', 

                                                   'answers_author_id', 'questions_title', 

                                                   'questions_body', 'answers_body', 

                                                   'questions_date_added', 'answers_date_added',

                                                   'professionals_location', 'professionals_industry',

                                                    'students_location']]
merge_qna_professionals_students.shape
merge_qna_professionals_students.head()