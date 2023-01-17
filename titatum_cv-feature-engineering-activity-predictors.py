import os

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
def collect_actors_activities(actors, actor_id):

    activities = pd.DataFrame(np.concatenate((actors[[actor_id]].merge(

                                                               questions[['questions_author_id', 'questions_date_added']], 

                                                               left_on=actor_id, right_on='questions_author_id', how='inner').values[:,[0,2]],

                                                           actors[[actor_id]].merge(

                                                               answers[['answers_author_id', 'answers_date_added']], 

                                                               left_on=actor_id, right_on='answers_author_id', how='inner').values[:,[0,2]],

                                                           actors[[actor_id]].merge(

                                                               comments[['comments_author_id', 'comments_date_added']], 

                                                               left_on=actor_id, right_on='comments_author_id', how='inner').values[:,[0,2]]), 

                                                          axis=0), 

                                           columns=[actor_id, 'activity_time'])

    activities['activity_date'] = activities['activity_time'].dt.date

    activities_df = activities.groupby(

        [actor_id, 'activity_date'])['activity_time'].count().reset_index().pivot(

        values='activity_time', columns=actor_id, index='activity_date')

    return activities_df



def compute_days_from_last_activities(activities, dates, ids):    

    days_from_last_activities = pd.DataFrame(1, index=dates, columns=ids)

    days_from_last_activities = days_from_last_activities.rolling(

        window=1000000, min_periods=1, center=False).sum()    

    activity_indicators = activities.reindex_like(days_from_last_activities).notnull()

    days_from_last_activities = days_from_last_activities.sub(

        days_from_last_activities[activity_indicators].fillna(method='ffill'))

    return days_from_last_activities
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
window_days = [100000, 365, 30]

professional_activities_df = collect_actors_activities(

    actors=professionals, actor_id='professionals_id')

professional_activities_df.index = pd.to_datetime(professional_activities_df.index)

for window in window_days:

    print('Process window: {}'.format(window))

    cum_sum_professional_activities = professional_activities_df.rolling(

        window=window, min_periods=1).sum()

    cum_sum_professional_activities.to_parquet(

        'professional_activities_sum_{}.parquet.gzip'.format(window), compression='gzip')

    cum_sum_professional_activities['e1d39b665987455fbcfbec3fc6df6056'].plot()
# Obtaining the lists of dates and professional ids #

professionals['professionals_joined_date'] = professionals['professionals_date_joined'].dt.date

dates = pd.date_range(start=professionals['professionals_joined_date'].min(), 

                      end=professionals['professionals_joined_date'].max())

ids = professionals['professionals_id']
days_from_last_activities = compute_days_from_last_activities(professional_activities_df, dates, ids)

days_from_last_activities.to_parquet(

    'days_from_last_activities.parquet.gzip', compression='gzip')
professionals['value'] = 1

professionals_joined_dates_df = professionals.pivot(

    values='value', index='professionals_joined_date', columns='professionals_id')

days_from_joined_dates = compute_days_from_last_activities(professionals_joined_dates_df, dates, ids)

days_from_joined_dates.to_parquet(

    'days_from_joined_dates.parquet.gzip', compression='gzip')
days_from_last_activities['ffca7b070c9d41e98eba01d23a920d52'].plot()
days_from_joined_dates['ffca7b070c9d41e98eba01d23a920d52'].plot()
os.listdir()