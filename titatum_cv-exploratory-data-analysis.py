import os

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
def generate_table_summary(data, data_name):

    print('The number of \'{}\' rows is {}'.format(data_name, data.shape[0]))

    print('')

    print('Column names and data types:')

    print(data.dtypes)



def generate_cum_plot(data, date_col, data_name):

    pd.DataFrame(1, index=data[date_col], columns=[data_name]).sort_index().cumsum().plot()

    plot_title = 'Cumulative Count of {} over Time'.format(data_name)

    plt.title(plot_title)

    plt.xlabel('Date')

    plt.xlabel('Cumulative Count')

    plt.savefig('{}.jpg'.format(plot_title.replace(' ', '').replace('/','')))

    plt.show()

    

def create_group_by_counts(data, 

                           group_by_col, group_by_name,

                           count_col, count_name, 

                           hist_cut_off, bin_numbers):

    group_by_counts = data.groupby(group_by_col)[count_col].count().sort_values(ascending=False)

    print('The number of unique \'{} by {}\' is {}'.format(count_name, group_by_name , len(group_by_counts)))

    print(group_by_counts.head(10))

    

    group_by_counts[group_by_counts < hist_cut_off].hist(bins=bin_numbers)

    plot_title = 'Histogram of {} by {}'.format(group_by_name, count_name)

    plt.title(plot_title)

    plt.xlabel(count_name)

    plt.ylabel(group_by_name)

    plt.savefig('{}.jpg'.format(plot_title.replace(' ', '').replace('/','')))

    plt.show()

    

    return group_by_counts



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



def plot_temporal_active_actors(activities_df, activity_days, actor_name):

    activities_df.rolling(window=activity_days, min_periods=1).sum().count(axis=1).plot()

    plot_title = '{}-day windowed Active {} over Time'.format(activity_days, actor_name)

    plt.title(plot_title)

    plt.xlabel('Date')

    plt.ylabel('Active {}'.format(actor_name))

    plt.savefig('{}.jpg'.format(plot_title.replace(' ', '').replace('/','')))

    plt.show()
# Input data files are available in the "../input/" directory.

input_dir = "../input"

print("\n".join(os.listdir(input_dir)))
# Load all data tables to explore #



career_village_dir = os.path.join(input_dir,'data-science-for-good-careervillage')



professionals = pd.read_csv(os.path.join(career_village_dir, 'professionals.csv'), parse_dates=True)

students = pd.read_csv(os.path.join(career_village_dir, 'students.csv'))

school_memberships = pd.read_csv(os.path.join(career_village_dir, 'school_memberships.csv'))

groups = pd.read_csv(os.path.join(career_village_dir, 'groups.csv'))

group_memberships = pd.read_csv(os.path.join(career_village_dir, 'group_memberships.csv'))

questions = pd.read_csv(os.path.join(career_village_dir, 'questions.csv'))

answers = pd.read_csv(os.path.join(career_village_dir, 'answers.csv'))

emails = pd.read_csv(os.path.join(career_village_dir, 'emails.csv'))

matches = pd.read_csv(os.path.join(career_village_dir, 'matches.csv'))

comments = pd.read_csv(os.path.join(career_village_dir, 'comments.csv'))

tags = pd.read_csv(os.path.join(career_village_dir, 'tags.csv'))

tag_users = pd.read_csv(os.path.join(career_village_dir, 'tag_users.csv'))

tag_questions = pd.read_csv(os.path.join(career_village_dir, 'tag_questions.csv'))



# Convert string dates to date time objects #



professionals['professionals_date_joined'] = pd.to_datetime(professionals['professionals_date_joined'])

students['students_date_joined'] = pd.to_datetime(students['students_date_joined'])

questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'])

answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'])

emails['emails_date_sent'] = pd.to_datetime(emails['emails_date_sent'])

comments['comments_date_added'] = pd.to_datetime(comments['comments_date_added'])
generate_table_summary(professionals, data_name='professionals')

professionals.sample(5)
generate_cum_plot(professionals, date_col='professionals_date_joined', data_name='Professionals')
professionals_location_counts = create_group_by_counts(data=professionals, 

                                                       group_by_col='professionals_location', group_by_name = 'Locations',

                                                       count_col='professionals_id', count_name='Professional Numbers',

                                                       hist_cut_off=50, bin_numbers=50)
professionals_industry_counts = create_group_by_counts(data=professionals, 

                                                       group_by_col='professionals_industry', group_by_name = 'Industries',

                                                       count_col='professionals_id', count_name='Professional Numbers',

                                                       hist_cut_off=50, bin_numbers=50)
professionals_headline_counts = create_group_by_counts(data=professionals, 

                                                       group_by_col='professionals_headline', group_by_name = 'Headlines',

                                                       count_col='professionals_id', count_name='Professional Numbers',

                                                       hist_cut_off=50, bin_numbers=50)
professional_activities_df = collect_actors_activities(actors=professionals, actor_id='professionals_id')

professional_activities_df.index = pd.to_datetime(professional_activities_df.index)

plot_temporal_active_actors(professional_activities_df, activity_days= 365, actor_name='Professionals')

plot_temporal_active_actors(professional_activities_df, activity_days= 30, actor_name='Professionals')

plot_temporal_active_actors(professional_activities_df, activity_days= 7, actor_name='Professionals')
generate_table_summary(students, data_name='students')

students.sample(5)
generate_cum_plot(students, date_col='students_date_joined', data_name='Students')
students_location_counts = create_group_by_counts(data=students, 

                                                  group_by_col='students_location', group_by_name = 'Locations',

                                                  count_col='students_id', count_name='Student Numbers',

                                                  hist_cut_off=50, bin_numbers=50)
print('The number of professionals in India is {}'.format(

    professionals.dropna(subset=['professionals_location']).apply(lambda row: row['professionals_location'].lower().find('india')>=0, axis=1).sum()))
student_activities_df = collect_actors_activities(actors=students, actor_id='students_id')

student_activities_df.index = pd.to_datetime(student_activities_df.index)

plot_temporal_active_actors(student_activities_df, activity_days= 365, actor_name='Students')

plot_temporal_active_actors(student_activities_df, activity_days= 30, actor_name='Students')

plot_temporal_active_actors(student_activities_df, activity_days= 7, actor_name='Students')
generate_table_summary(school_memberships, data_name='school memberships')

school_memberships.head(3)
school_memberships_counts = create_group_by_counts(data=school_memberships, 

                                                   group_by_col='school_memberships_school_id', group_by_name = 'Schools',

                                                   count_col='school_memberships_user_id', count_name='User Numbers',

                                                   hist_cut_off=50, bin_numbers=50)
generate_table_summary(groups, data_name='groups')

groups.head(3)
group_type_counts = create_group_by_counts(data=groups, 

                                           group_by_col='groups_group_type', group_by_name = 'Group Types',

                                           count_col='groups_id', count_name='Group Numbers',

                                           hist_cut_off=50, bin_numbers=50)
generate_table_summary(group_memberships, data_name='group memberships')

group_memberships.head(3)
group_memberships_counts = create_group_by_counts(data=group_memberships, 

                                                  group_by_col='group_memberships_group_id', group_by_name = 'Groups',

                                                  count_col='group_memberships_user_id', count_name='User Numbers',

                                                  hist_cut_off=200, bin_numbers=100)
generate_table_summary(questions, data_name='questions')

questions.sample(3)
generate_cum_plot(questions, date_col='questions_date_added', data_name='Questions')
print('The average number of questions that a student asks on the website is {}'.format(round(questions.groupby('questions_author_id')['questions_id'].count().mean(), 0)))

questions_authors_counts = create_group_by_counts(data=questions, 

                                                  group_by_col='questions_author_id', group_by_name = 'Users',

                                                  count_col='questions_id', count_name='Question Numbers',

                                                  hist_cut_off=200, bin_numbers=100)
generate_table_summary(answers, data_name='answers')

answers.sample(3)
generate_cum_plot(answers, date_col='answers_date_added', data_name='Answers')
print('The average number of answers for each contributing professional {}'.format(round(answers.groupby('answers_author_id')['answers_id'].count().mean(), 0)))

answers_authors_counts = create_group_by_counts(data=answers, 

                                                  group_by_col='answers_author_id', group_by_name = 'Users',

                                                  count_col='answers_id', count_name='Answer Numbers',

                                                  hist_cut_off=200, bin_numbers=100)
print('The average number of answers for each responsed question {}'.format(round(answers.groupby('answers_question_id')['answers_id'].count().mean(), 0)))

answers_questions_counts = create_group_by_counts(data=answers, 

                                                  group_by_col='answers_question_id', group_by_name = 'Questions',

                                                  count_col='answers_id', count_name='Answer Numbers',

                                                  hist_cut_off=200, bin_numbers=100)

answers_questions_counts.describe()
answers_by_professionalss = answers.merge(professionals, left_on='answers_author_id', right_on='professionals_id', how='inner')

print('The percentage of answers from professionals is {}'.format(round(100.0 * answers_by_professionalss.shape[0] / answers.shape[0], 2)))

answers_by_students = answers.merge(students, left_on='answers_author_id', right_on='students_id', how='inner')

print('The percentage of answers from students is {}'.format(round(100.0 * answers_by_students.shape[0] / answers.shape[0], 2)))
snap_shot_date = dt.datetime(2018, 7, 1)

print('The snapshot date is {}'.format(snap_shot_date))



questions_july_2018 = questions[questions['questions_date_added'] >= snap_shot_date]

print('The number of questions after the snap shot date is {}'.format(questions_july_2018.shape[0]))

print('Compared to the full data set, the percentage of questions after the snap shot date is {}%'.format(round((100.0 * questions_july_2018.shape[0]) / questions.shape[0], 2)))



answers_july_2018 = answers[answers['answers_date_added'] >= snap_shot_date]

print('The number of answers after the snap shot date is {}'.format(answers_july_2018.shape[0]))

print('Compared to the full data set, the percentage of answers after the snap shot date is {}%'.format(round((100.0 * answers_july_2018.shape[0]) / answers.shape[0], 2)))
professional_activities_df_before_july_2018_df = professional_activities_df.iloc[:professional_activities_df.index.get_loc(snap_shot_date)-1]

professional_activities_df_before_july_2018 = pd.DataFrame(

    professional_activities_df_before_july_2018_df.sum(axis=0).values, index=professional_activities_df_before_july_2018_df.columns, columns=['100000_day_activity_count'])

professional_activities_df_before_july_2018['365_day_activity_count'] = professional_activities_df_before_july_2018_df.iloc[-365:].sum(axis=0)

professional_activities_df_before_july_2018['30_day_activity_count'] = professional_activities_df_before_july_2018_df.iloc[-30:].sum(axis=0)

professional_activities_df_before_july_2018.dropna().sort_values(by='100000_day_activity_count', ascending=False).head(10)



activities_answers_july_2018 = answers_july_2018.merge(professional_activities_df_before_july_2018.reset_index(), left_on='answers_author_id', right_on='professionals_id', how='outer')

professionals_with_answers_july_2018 = activities_answers_july_2018.dropna(subset=['answers_id'])[professional_activities_df_before_july_2018.columns]

print('Activities of Professionals with Answers:\n{}'.format(professionals_with_answers_july_2018.mean()))

print('Activities of Professionals Overall:\n{}'.format(professional_activities_df_before_july_2018.mean()))
generate_table_summary(emails, data_name='emails')

emails.sample(5)
generate_cum_plot(emails, date_col='emails_date_sent', data_name='Emails')
emails.groupby(['emails_frequency_level'])['emails_id'].count().plot.bar()

plt.savefig('volumes_of_each_emails_frequency_level.jpg')
emails_recipients_counts = create_group_by_counts(data=emails, 

                                                  group_by_col='emails_recipient_id', group_by_name = 'Users',

                                                  count_col='emails_id', count_name='Email Numbers',

                                                  hist_cut_off=200, bin_numbers=100)
generate_table_summary(matches, data_name='matches')

matches.head(3)
matches_emails_counts = create_group_by_counts(data=matches, 

                                                  group_by_col='matches_email_id', group_by_name = 'Emails',

                                                  count_col='matches_question_id', count_name='Question Numbers',

                                                  hist_cut_off=200, bin_numbers=100)

print('The average number of recommended questions in each email is {}'.format(round(matches_emails_counts.mean(),0)))

print('The number of emails that have at most 3 recommendations is {}'.format(matches_emails_counts[matches_emails_counts <=3].shape[0]))
matches_questions_counts = create_group_by_counts(data=matches, 

                                                  group_by_col='matches_question_id', group_by_name = 'Questions',

                                                  count_col='matches_email_id', count_name='Email Numbers',

                                                  hist_cut_off=200, bin_numbers=100)

print('The average number of emails sent for each question is {}'.format(round(matches_questions_counts.mean(),0)))
ml_data_dir = os.path.join(input_dir,'cv-machine-learning-data-construction')

examples = pd.read_parquet(os.path.join(ml_data_dir,'positive_negative_examples.parquet.gzip'))

examples['emails_date'] = examples['emails_date_sent'].dt.date

matches_per_email = examples.groupby(['answer_user_id', 'emails_date'])['questions_id'].count().reset_index()
print('The average number of questions that a professional could receive on an active day is {}'.format(round(matches_per_email['questions_id'].mean())))

matches_per_email[matches_per_email['questions_id'] <= 50]['questions_id'].hist(bins=50)

plot_title = 'The distribution of questions in emails per professional in an active day'

plt.title(plot_title)

plt.xlabel('Number of questions on an active day')

plt.xlabel('Cumulative Count')

plt.savefig('questions_in_emails_per professional_in_an_active_day.jpg')

plt.show()
generate_table_summary(comments, data_name='comments')

comments.sample(5)
generate_cum_plot(comments, date_col='comments_date_added', data_name='Comments')
comments_authors_counts = create_group_by_counts(data=comments, 

                                                  group_by_col='comments_author_id', group_by_name = 'Users',

                                                  count_col='comments_id', count_name='Comment Numbers',

                                                  hist_cut_off=200, bin_numbers=100)

comments_authors_counts.describe()
comments_qa_counts = create_group_by_counts(data=comments, 

                                                  group_by_col='comments_parent_content_id', group_by_name = 'Questions/Answers',

                                                  count_col='comments_id', count_name='Comment Numbers',

                                                  hist_cut_off=200, bin_numbers=100)

comments_qa_counts.describe()
generate_table_summary(tags, data_name='tags')

tags.sample(10)
tags.merge(tag_questions, left_on='tags_tag_id', right_on='tag_questions_tag_id').groupby(

    'tags_tag_name')['tag_questions_question_id'].count().sort_values(ascending=False).head(10)
tags.merge(tag_users, left_on='tags_tag_id', right_on='tag_users_tag_id').groupby(

    'tags_tag_name')['tag_users_user_id'].count().sort_values(ascending=False).head(10)
generate_table_summary(tag_users, data_name='tag users')

tag_users.head(3)
tags_users_counts = create_group_by_counts(data=tag_users, 

                                           group_by_col='tag_users_user_id', group_by_name = 'Users',

                                           count_col='tag_users_tag_id', count_name='Tag Numbers',

                                           hist_cut_off=200, bin_numbers=100)

tags_users_counts.describe()

print('The number of tag registering users having at most 3 tags is {}'.format(sum(tags_users_counts<=3)))
professionals.merge(tags_users_counts.reset_index(), left_on='professionals_id', right_on='tag_users_user_id', how='inner').shape[0]
students.merge(tags_users_counts.reset_index(), left_on='students_id', right_on='tag_users_user_id', how='inner').shape[0]
generate_table_summary(tag_questions, data_name='tag questions')

tag_questions.head(3)
tag_questions_counts = create_group_by_counts(data=tag_questions, 

                                                  group_by_col='tag_questions_question_id', group_by_name = 'Questions',

                                                  count_col='tag_questions_tag_id', count_name='Tag Numbers',

                                                  hist_cut_off=200, bin_numbers=100)

tag_questions_counts.describe()