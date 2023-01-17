import os

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.

input_dir = "../input"



professionals = pd.read_csv(os.path.join(input_dir, 'professionals.csv'), parse_dates=True)

students = pd.read_csv(os.path.join(input_dir, 'students.csv'))

school_memberships = pd.read_csv(os.path.join(input_dir, 'school_memberships.csv'))

groups = pd.read_csv(os.path.join(input_dir, 'groups.csv'))

group_memberships = pd.read_csv(os.path.join(input_dir, 'group_memberships.csv'))

questions = pd.read_csv(os.path.join(input_dir, 'questions.csv'))

answers = pd.read_csv(os.path.join(input_dir, 'answers.csv'))

emails = pd.read_csv(os.path.join(input_dir, 'emails.csv'))

matches = pd.read_csv(os.path.join(input_dir, 'matches.csv'))

comments = pd.read_csv(os.path.join(input_dir, 'comments.csv'))

tags = pd.read_csv(os.path.join(input_dir, 'tags.csv'))

tag_users = pd.read_csv(os.path.join(input_dir, 'tag_users.csv'))

tag_questions = pd.read_csv(os.path.join(input_dir, 'tag_questions.csv'))



professionals['professionals_date_joined'] = pd.to_datetime(professionals['professionals_date_joined'])

students['students_date_joined'] = pd.to_datetime(students['students_date_joined'])

questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'])

answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'])

emails['emails_date_sent'] = pd.to_datetime(emails['emails_date_sent'])

comments['comments_date_added'] = pd.to_datetime(comments['comments_date_added'])
questions_users_shared_tags = tag_questions.merge(

    tag_users, left_on='tag_questions_tag_id', right_on='tag_users_tag_id', 

    how='inner').drop_duplicates().groupby(

    ['tag_questions_question_id', 'tag_users_user_id'])['tag_questions_tag_id'].count().reset_index()

questions_users_shared_tags = questions_users_shared_tags.rename(

    columns = {'tag_questions_question_id': 'questions_id', 

               'tag_users_user_id': 'answer_user_id', 

               'tag_questions_tag_id': 'question_user_shared_tags'})
print(questions_users_shared_tags.shape)

questions_users_shared_tags.sort_values(by='question_user_shared_tags', ascending=False).head(3)

questions_users_shared_tags.to_parquet(

    'questions_users_shared_tags.parquet.gzip', compression='gzip')
users_users_shared_tags = tag_users.merge(tag_users, on='tag_users_tag_id')

users_users_shared_tags = users_users_shared_tags[

    users_users_shared_tags['tag_users_user_id_x'] != 

    users_users_shared_tags['tag_users_user_id_y']].drop_duplicates()

users_users_shared_tags = users_users_shared_tags.groupby(

    ['tag_users_user_id_x', 'tag_users_user_id_y'])['tag_users_tag_id'].count().reset_index()

users_users_shared_tags = users_users_shared_tags.rename(

    columns = {'tag_users_user_id_x': 'questions_author_id',

               'tag_users_user_id_y': 'answer_user_id',

               'tag_users_tag_id': 'questioner_answerer_shared_tags'})
print(users_users_shared_tags.shape)

users_users_shared_tags.sort_values(by='questioner_answerer_shared_tags', ascending=False).head(3)

users_users_shared_tags.to_parquet(

    'users_users_shared_tags.parquet.gzip', compression='gzip')
users_users_shared_groups = group_memberships.merge(

    group_memberships, on='group_memberships_group_id')

users_users_shared_groups = users_users_shared_groups[

    users_users_shared_groups['group_memberships_user_id_x'] != 

    users_users_shared_groups['group_memberships_user_id_y']].drop_duplicates()

users_users_shared_groups = users_users_shared_groups.groupby(

    ['group_memberships_user_id_x', 'group_memberships_user_id_y'])[

    'group_memberships_group_id'].count().reset_index()

users_users_shared_groups = users_users_shared_groups.rename(

    columns = {'group_memberships_user_id_x': 'questions_author_id',

               'group_memberships_user_id_y': 'answer_user_id',

               'group_memberships_group_id': 'questioner_answerer_shared_groups'})
print(users_users_shared_groups.shape)

users_users_shared_groups.sort_values(by='questioner_answerer_shared_groups', ascending=False).tail(3)

users_users_shared_groups.to_parquet(

    'users_users_shared_groups.parquet.gzip', compression='gzip')
users_users_shared_schools = school_memberships.merge(

    school_memberships, on='school_memberships_school_id')

users_users_shared_schools = users_users_shared_schools[

    users_users_shared_schools['school_memberships_user_id_x'] != 

    users_users_shared_schools['school_memberships_user_id_y']].drop_duplicates()

users_users_shared_schools = users_users_shared_schools.groupby(

    ['school_memberships_user_id_x', 'school_memberships_user_id_y'])[

    'school_memberships_school_id'].count().reset_index()

users_users_shared_schools = users_users_shared_schools.rename(

    columns = {'school_memberships_user_id_x': 'questions_author_id',

               'school_memberships_user_id_y': 'answer_user_id',

               'school_memberships_school_id': 'questioner_answerer_shared_schools'})
print(users_users_shared_schools.shape)

users_users_shared_schools.sort_values(by='questioner_answerer_shared_schools', ascending=False).head(3)

users_users_shared_schools.to_parquet(

    'users_users_shared_schools.parquet.gzip', compression='gzip')
os.listdir()