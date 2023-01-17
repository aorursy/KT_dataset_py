import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

root_path = '../input/'

print('The csv files provided are:\n')

print(os.listdir(root_path))
df_emails = pd.read_csv(root_path + 'emails.csv')

df_questions = pd.read_csv(root_path + 'questions.csv')

df_professionals = pd.read_csv(root_path + 'professionals.csv')

df_comments = pd.read_csv(root_path + 'comments.csv')

df_tag_users = pd.read_csv(root_path + 'tag_users.csv')

df_group_memberships = pd.read_csv(root_path + 'group_memberships.csv')

df_tags = pd.read_csv(root_path + 'tags.csv')

df_answer_scores = pd.read_csv(root_path + 'answer_scores.csv')

df_students = pd.read_csv(root_path + 'students.csv')

df_groups = pd.read_csv(root_path + 'groups.csv')

df_tag_questions = pd.read_csv(root_path + 'tag_questions.csv')

df_question_scores = pd.read_csv(root_path + 'question_scores.csv')

df_matches = pd.read_csv(root_path + 'matches.csv')

df_answers = pd.read_csv(root_path + 'answers.csv')

df_school_memberships = pd.read_csv(root_path + 'school_memberships.csv')
print('questions')

display(df_questions.head(2))

print('answers')

display(df_answers.head(2))
df_questions['questions_date_added'] = pd.to_datetime(df_questions['questions_date_added'])

df_answers['answers_date_added'] = pd.to_datetime(df_answers['answers_date_added'])

df_qa = pd.merge(df_questions, df_answers, left_on='questions_id', right_on='answers_question_id', how='left')

df_qa_grouped = df_qa.groupby('questions_id').agg({'questions_date_added':min, 'answers_date_added':min,

                                                   'questions_body':min})

df_qa_grouped['days_taken'] = (df_qa_grouped['answers_date_added'] - df_qa_grouped['questions_date_added']).dt.days

df_qa_grouped['questions_body_length'] = df_qa_grouped['questions_body'].apply(len)
print('Numerical summary of days taken to answer a question')

display(df_qa_grouped['days_taken'].describe())

plt.figure(figsize=(10,6))

plt.title('Distribution of days taken to answer a question')

plt.hist(df_qa_grouped['days_taken'], color='blue', edgecolor='black', bins=100)

plt.xlabel('min days taken to answer a question')

plt.ylabel('count')
print('Correlation between length of the body of questions and response time')

display(df_qa_grouped[['questions_body_length', 'days_taken']].corr())

plt.figure(figsize=(10,6))

plt.scatter(df_qa_grouped['questions_body_length'], df_qa_grouped['days_taken'])

plt.xlabel('questions_body_length')

plt.ylabel('days_taken')
print('The number of questions that are unanswered are:', df_qa['answers_id'].isnull().sum(axis=0))
print('Numerical summary of count of questions with respect to time')

display(df_questions['questions_date_added'].dt.year.describe())

plt.figure(figsize=(10,6))

plt.title('Count of questions with respect to time')

sns.countplot(df_questions['questions_date_added'].dt.year, color='violet')
print('tags')

display(df_tags.head(2))

print('tag_users')

display(df_tag_users.head(2))

print('tag_questions')

display(df_tag_questions.head(2))
#To see the tags that every user follows 

df_tag_users_merged = pd.merge(df_tag_users, df_tags, left_on='tag_users_tag_id', right_on='tags_tag_id', how='inner')

#To see the tags that are linked with every question

df_tag_questions_merged = pd.merge(df_tag_questions, df_tags, left_on='tag_questions_tag_id', right_on='tags_tag_id', how='inner')
plt.figure(figsize=(10,6))

plt.title('50 most popular tags wrt user following')

sns.countplot(df_tag_users_merged[df_tag_users_merged['tags_tag_name'].isin(

    df_tag_users_merged['tags_tag_name'].value_counts().index[:50])]['tags_tag_name'], color='maroon', order=df_tag_users_merged['tags_tag_name'].value_counts().index[:50])

plt.ylabel('count')

plt.xticks(rotation='vertical')
plt.figure(figsize=(10,6))

plt.title('50 most popular tags wrt the number of questions they are linked to')

sns.countplot(df_tag_questions_merged[df_tag_questions_merged['tags_tag_name'].isin(

    df_tag_questions_merged['tags_tag_name'].value_counts().index[:50])]['tags_tag_name'], color='maroon', order=df_tag_questions_merged['tags_tag_name'].value_counts().index[:50])

plt.ylabel('count')

plt.xticks(rotation='vertical')
relevant_tags = set(df_tag_questions_merged['tag_questions_tag_id'].unique()).union(set(df_tag_users_merged['tag_users_tag_id'].unique()))

len(relevant_tags)
print('The total number of unique tagged users is:', df_tag_users_merged['tag_users_user_id'].nunique())

print('The total number of unique tags is:', df_tags['tags_tag_id'].nunique())

print('The total number of unique tagged questions is:', df_tag_questions_merged['tag_questions_question_id'].nunique())

print('The proportion of total questions that are linked with tags:', 

      df_tag_questions_merged['tag_questions_question_id'].nunique()/df_questions['questions_id'].nunique())

print('The proportion of tags that are linked with questions out of the total number of tags:', 

      df_tag_questions_merged['tag_questions_tag_id'].nunique()/df_tags['tags_tag_id'].nunique())

print('The proportion of tags that are followed by users out of the total number of tags:', 

      df_tag_users_merged['tag_users_tag_id'].nunique()/df_tags['tags_tag_id'].nunique())

print('The total number of tags that have a user following > 1% :', 

      sum(df_tag_users_merged['tags_tag_name'].value_counts()/df_tag_users_merged['tag_users_user_id'].nunique() > 0.01)) 

print('The total number of tags that are used in > 1% of the tagged questions:', 

      sum(df_tag_questions_merged['tags_tag_name'].value_counts()/df_tag_questions_merged['tag_questions_question_id'].nunique() > 0.01)) 
user_tags = list((df_tag_users_merged['tags_tag_name'].value_counts()/df_tag_users_merged['tag_users_user_id'].nunique() 

                     > 0.01).index[(df_tag_users_merged['tags_tag_name'].value_counts()/df_tag_users_merged['tag_users_user_id'].nunique() > 0.01)])

question_tags = list((df_tag_questions_merged['tags_tag_name'].value_counts()/df_tag_questions_merged['tag_questions_question_id'].nunique() 

                     > 0.01).index[(df_tag_questions_merged['tags_tag_name'].value_counts()/df_tag_questions_merged['tag_questions_question_id'].nunique() > 0.01)])

print('The total number of tags:', len(set(question_tags).union(user_tags)))

print('The number of common tags:', len(set(question_tags).intersection(user_tags)))

print('The tags are:\n', set(question_tags).intersection(user_tags))
print('Coverage of tagged questions:', df_tag_questions_merged[df_tag_questions_merged['tags_tag_name'].isin(

    set(user_tags).union(set(question_tags)))]['tag_questions_question_id'].nunique()/df_tag_questions_merged['tag_questions_question_id'].nunique())
print('Coverage of tagged users:', df_tag_users_merged[df_tag_users_merged['tags_tag_name'].isin(

    set(user_tags).union(set(question_tags)))]['tag_users_user_id'].nunique()/df_tag_users_merged['tag_users_user_id'].nunique())
def print_if_found(df, column, string):

    print(df[df[column].str.contains(string)][column].unique())

print_if_found(df_tag_users_merged, 'tags_tag_name', 'computer')
print_if_found(df_tag_users_merged, 'tags_tag_name', 'psychology')
#Looking at tags and questions

print('Numerical summary of the number of tags linked with every question')

display(df_tag_questions.groupby('tag_questions_question_id').agg({'tag_questions_tag_id':'count'})['tag_questions_tag_id'].describe())

plt.figure(figsize=(10,6))

plt.title('Count of the number of tags linked with every question')

sns.countplot(df_tag_questions.groupby('tag_questions_question_id').agg({'tag_questions_tag_id':'count'})['tag_questions_tag_id'], color='orange')

plt.xlabel('count of tags')

plt.ylabel('count')
print('professionals')

display(df_professionals.head(2))
#To see the profile of the volunteers and the questions that they have answered

df_answers_professionals = pd.merge(df_answers, df_professionals, left_on='answers_author_id', right_on='professionals_id', how='outer')
print('Number of professionals that are there on the platform:', df_professionals['professionals_id'].nunique())

print('Number of professionals that haven\'t answered questions on the platform:', df_answers_professionals['answers_id'].isnull().sum())

print('Number of answers that have been answered by users who have changed their registration type:', 

      len(set(df_answers['answers_author_id']) - set(df_professionals['professionals_id'])))

print('Proportion of professionals who haven\'t answered a question:', 

     df_answers_professionals['answers_id'].isnull().sum()/df_professionals['professionals_id'].nunique())
last_date = df_questions['questions_date_added'].max() #date of the last question asked on the platform

df_ap_grouped = df_answers_professionals.groupby('professionals_id').agg({'answers_date_added':max}).apply(lambda x:

                                                                                          (last_date-x).dt.days)

df_ap_grouped.rename(columns={'answers_date_added':'days_since_answered'}, inplace=True)

print('Numerical summary of days_since_answered')

display(df_ap_grouped['days_since_answered'].describe())

plt.figure(figsize=(10,6))

plt.title('Activity of professionals')

plt.hist(df_ap_grouped['days_since_answered'], bins=50, color='blue', edgecolor='black')

plt.xlabel('days_since_answered')

plt.ylabel('count')
plt.figure(figsize=(10,6))

plt.title('Count of years since last answered')

sns.countplot((df_ap_grouped[pd.notnull(df_ap_grouped['days_since_answered'])]['days_since_answered']/365).apply(round), color='magenta')
#Looking at professionals and tag

df_tag_professionals = pd.merge(df_tag_users_merged, df_professionals, left_on='tag_users_user_id', 

                                 right_on='professionals_id')

print('Proportion of tagged users that are professionals:', df_tag_professionals['professionals_id'].nunique()/df_tag_users_merged['tag_users_user_id'].nunique())
print('Numerical summary of number of tags followed by tagged professionals')

display(df_tag_professionals.groupby('professionals_id').agg({'tag_users_tag_id':lambda x: len(x)})['tag_users_tag_id'].describe())

plt.figure(figsize=(10,6))

plt.title('Count of number of tags followed by tagged professionals')

sns.countplot(df_tag_professionals.groupby('professionals_id').agg({'tag_users_tag_id':len})['tag_users_tag_id'].astype(int), color='cyan')

plt.xlim((0,30))
print('emails')

display(df_emails.head(2))

print('matches')

display(df_matches.head(2))
df_emails_matches = pd.merge(df_emails, df_matches, left_on='emails_id', right_on='matches_email_id')

df_em_answers = pd.merge(df_emails_matches, df_answers, left_on=['emails_recipient_id','matches_question_id'], 

                              right_on=['answers_author_id', 'answers_question_id'], how='left')

grouped_response_rate = df_em_answers.groupby('emails_recipient_id').agg({'answers_id':lambda x: 

                                   (x.notnull().sum())/len(x)})['answers_id']

df_em_answers.head(2)
print('Number of professionals that have been sent emails:', df_emails_matches['emails_recipient_id'].nunique())

print('Number of professionals that have not answered even one question that was emailed to them:',

      grouped_response_rate[grouped_response_rate == 0].shape[0])
df_em_answers_right = pd.merge(df_emails_matches, df_answers, left_on=['emails_recipient_id','matches_question_id'], 

                              right_on=['answers_author_id', 'answers_question_id'], how='right')

print('Number of professionals who have answered questions that weren\'t emailed to them:', 

     (df_em_answers_right.groupby('answers_author_id').agg({'emails_id': lambda x: x.isnull().sum()})['emails_id']!=0).sum())
print('Numerical summary of the response rate')

display(grouped_response_rate.describe())

plt.figure(figsize=(10,6))

plt.title('Response rate of matched questions')

plt.hist(grouped_response_rate, color='blue', edgecolor='black',bins=50)

plt.ylabel('count')

plt.xlabel('response rate')
plt.figure(figsize=(10,6))

plt.title('Response rate of matched questions > 0')

plt.hist(grouped_response_rate[grouped_response_rate > 0], color='blue', edgecolor='black',bins=50)

plt.ylabel('count')

plt.xlabel('response rate > 0')
grouped_immediate_rate = df_emails_matches.drop_duplicates(['emails_recipient_id', 'emails_id']).groupby('emails_recipient_id').agg({'emails_frequency_level': 

                                        lambda x: (x.str.contains('email_notification_immediate').sum())/len(x)})['emails_frequency_level']

plt.figure(figsize=(10,6))

plt.title('Proportion of immediate emails')

plt.hist(grouped_immediate_rate, color='blue', edgecolor='black', bins=50)

plt.ylabel('count')

plt.xlabel('proportion of immmediate emails')
grouped_immediate_response = df_em_answers.groupby(['emails_recipient_id', 'emails_frequency_level']).agg({'answers_id': 

                                                    lambda x: (x.notnull().sum())/len(x)}).reset_index()

plt.figure(figsize=(10,6))

plt.title('Distribution of response rate of immediate questions')

plt.hist(grouped_immediate_response[grouped_immediate_response['emails_frequency_level']==

                                    'email_notification_immediate']['answers_id'], color='blue', edgecolor='black', bins=50)

plt.xlabel('response rate')

plt.ylabel('count')
print('students')

display(df_students.head(2))

print('groups')

display(df_groups.head(2))

print('group_memberships')

display(df_group_memberships.head(2))

print('school_memberships')

display(df_school_memberships.head(2))
#To see the group memberships and type together

df_groups_merged = pd.merge(df_group_memberships, df_groups, left_on='group_memberships_group_id', right_on='groups_id', how='outer')

df_groups_professionals = pd.merge(df_groups_merged, df_professionals, left_on='group_memberships_user_id', right_on='professionals_id')

df_groups_students = pd.merge(df_groups_merged, df_students, left_on='group_memberships_user_id', right_on='students_id')

df_school_professionals = pd.merge(df_school_memberships, df_professionals, left_on='school_memberships_user_id', right_on='professionals_id')

df_school_students = pd.merge(df_school_memberships, df_students, left_on='school_memberships_user_id', right_on='students_id')
print('Number of groups that don\'t have a user following:', df_groups_merged['group_memberships_group_id'].isnull().sum())

print('Total number of users that have a group membership:', df_groups_merged['group_memberships_user_id'].nunique())

print('Proportion of users in the group memberships that are professionals:', df_groups_professionals['professionals_id'].nunique()/

     df_groups_merged['group_memberships_user_id'].nunique())

print('Proportion of users in the group memberships that are students:', df_groups_students['students_id'].nunique()/

     df_groups_merged['group_memberships_user_id'].nunique())

print('Total number of users that have a school membership:', df_school_memberships['school_memberships_user_id'].nunique())

print('Proportion of users in the school memberships that are professionals:', df_school_professionals['professionals_id'].nunique()/

     df_school_memberships['school_memberships_user_id'].nunique())

print('Proportion of users in the school memberships that are students:', df_school_students['students_id'].nunique()/

     df_school_memberships['school_memberships_user_id'].nunique())
df_students_questions = pd.merge(df_students, df_questions, left_on='students_id', right_on='questions_author_id')

print('Number of students on the platform:', df_students['students_id'].nunique())

print('Proportion of students who have asked a question on the platform:', df_students_questions['students_id'].nunique()/df_students['students_id'].nunique())
print('comments')

display(df_comments.head(2))

print('question_scores')

display(df_question_scores.head(2))

print('answer_scores')

display(df_answer_scores.head(2))
print('Number of comments:', df_comments['comments_id'].nunique())

print('Proportion of comments on questions:',

      len(set(df_comments['comments_parent_content_id']).intersection(set(df_questions['questions_id'])))/df_comments['comments_parent_content_id'].nunique())

print('Proportion of comments on answers:',

      len(set(df_comments['comments_parent_content_id']).intersection(set(df_answers['answers_id'])))/df_comments['comments_parent_content_id'].nunique())
print('Numerical summary of count of hearts on questions')

display(df_question_scores['score'].describe())

plt.figure(figsize=(10,6))

plt.title('Count of hearts on questions ')

sns.countplot(df_question_scores['score'], color='red')

plt.xlim(0,30)
print('Numerical summary of count of hearts on answers')

display(df_answer_scores['score'].describe())

plt.figure(figsize=(10,6))

plt.title('Count of hearts on answers ')

sns.countplot(df_answer_scores['score'], color='red')