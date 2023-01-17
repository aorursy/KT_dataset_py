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
def compute_3_hop_paths(left_table, right_table, 

                        left_on, right_on,

                        left_user, right_user,

                        left_date, right_date, 

                        path_name, left_name, right_name,

                        reflected = False):

    users_paths = left_table.merge(right_table, left_on=left_on, right_on=right_on, how='inner')[[

        left_user, left_date, right_user, right_date]]

    users_paths = users_paths[users_paths[left_user] != users_paths[right_user]]

    users_paths['date_created'] = users_paths[[left_date, right_date]].max(axis=1).dt.date

    users_paths = users_paths.groupby(['date_created', left_user, right_user])[left_date].count().reset_index()

    users_paths = users_paths.rename(columns={left_date: path_name})

    users_paths['left_right'] = users_paths[left_user] +  '-' + users_paths[right_user]

    users_paths = users_paths.pivot(values=path_name, columns='left_right', index='date_created')



    cum_sum_users_paths = users_paths.rolling(window=100000, min_periods=1).sum().stack().reset_index()

    cum_sum_users_paths[left_name] = cum_sum_users_paths['left_right'].apply(lambda x: x.split('-')[0])

    cum_sum_users_paths[right_name] = cum_sum_users_paths['left_right'].apply(lambda x: x.split('-')[1])

    del cum_sum_users_paths['left_right']

    cum_sum_users_paths = cum_sum_users_paths.rename(columns={0: path_name})

    

    if reflected:

        cum_sum_users_paths = pd.concat([cum_sum_users_paths,

                                      cum_sum_users_paths[cum_sum_users_paths.columns[[0,1,3,2]]].rename(

                                          columns={cum_sum_users_paths.columns[2]:cum_sum_users_paths.columns[3],

                                                   cum_sum_users_paths.columns[3]:cum_sum_users_paths.columns[2]})], axis=0)

        cum_sum_users_paths = cum_sum_users_paths.drop_duplicates()

    

    return cum_sum_users_paths
questioners_answerers_paths = compute_3_hop_paths(

    left_table = questions, right_table = answers, 

    left_on = 'questions_id', right_on = 'answers_question_id',

    left_user = 'questions_author_id', right_user = 'answers_author_id', 

    left_date = 'questions_date_added', right_date = 'answers_date_added',

    path_name = 'questioners_answerers_paths', left_name = 'questioner_id', right_name = 'answerer_id',

    reflected = False)

questioners_answerers_paths.sort_values(by='questioners_answerers_paths', ascending=False).head(5)

questioners_answerers_paths.to_parquet(

    'questioners_answerers_paths.parquet.gzip', compression='gzip')
commenters_questioners_paths = compute_3_hop_paths(

    left_table = comments, right_table = questions, 

    left_on = 'comments_parent_content_id', right_on = 'questions_id',

    left_user = 'comments_author_id', right_user = 'questions_author_id', 

    left_date = 'comments_date_added', right_date = 'questions_date_added',

    path_name = 'commenters_questioners_paths', left_name = 'commenter_id', right_name = 'questioner_id',

    reflected = False)

commenters_questioners_paths.sort_values(by='commenters_questioners_paths', ascending=False).head(5)

commenters_questioners_paths.to_parquet(

    'commenters_questioners_paths.parquet.gzip', compression='gzip')
commenters_answerers_paths = compute_3_hop_paths(

    left_table = comments, right_table = answers, 

    left_on = 'comments_parent_content_id', right_on = 'answers_question_id',

    left_user = 'comments_author_id', right_user = 'answers_author_id', 

    left_date = 'comments_date_added', right_date = 'answers_date_added',

    path_name = 'commenters_answerers_paths', left_name = 'commenter_id', right_name = 'answerer_id',

    reflected = False)

commenters_answerers_paths.sort_values(by='commenters_answerers_paths', ascending=False).head(5)

commenters_answerers_paths.to_parquet(

    'commenters_answerers_paths.parquet.gzip', compression='gzip')
os.listdir()