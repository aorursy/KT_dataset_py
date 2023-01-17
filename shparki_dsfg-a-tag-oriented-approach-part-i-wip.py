# Import required packages



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from pandas.api.types import CategoricalDtype

from IPython.display import display
answers = pd.read_csv('../input/answers.csv',

                           infer_datetime_format = True,

                           dtype = {

                              'answers_id': str,

                              'answers_author_id': str,

                              'answers_question_id': str,

                              'answers_body': str

                          },

                          parse_dates = ['answers_date_added'])

answers.columns = ['id', 'author_id', 'question_id', 'date_added', 'body']

answers.set_index('id', inplace = True)

answers.sort_index(kind = 'mergesort', inplace = True)

display(answers.head())
comments = pd.read_csv('../input/comments.csv',

                           infer_datetime_format = True,

                           dtype = {

                               'comments_id': str,

                               'comments_author_id': str,

                               'comments_parent_content_id': str,

                               'comments_body': str

                           },

                           parse_dates = ['comments_date_added'])

comments.columns = ['id', 'author_id', 'parent_id', 'date_added', 'body']

comments.set_index('id', inplace = True)

comments.sort_index(kind = 'mergesort', inplace = True)

display(comments.head())
# Look into categories for email_freq



_cat_emails = pd.read_csv('../input/emails.csv', 

                         usecols=['emails_frequency_level'],

                        squeeze = True).unique()

print(_cat_emails)
cat_emails = CategoricalDtype(

    categories = [ 'email_notification_immediate', 'email_notification_daily',

                   'email_notification_weekly'],

     ordered=True) # Not using _cat_emails because ordered to preserve frequency order
emails = pd.read_csv('../input/emails.csv',

                         infer_datetime_format = True,

                         dtype = {

                             'emails_id': str,

                             'emails_recipient_id': str,

                             'emails_frequency_level': cat_emails,

                         },

                         parse_dates = ['emails_date_sent'])

emails.columns = ['id', 'recipient_id', 'date_sent', 'frequency']

emails.set_index('id', inplace=True)

emails.sort_index(kind = 'mergesort', inplace = True)

display(emails.head())
group_memberships = pd.read_csv('../input/group_memberships.csv',

                                    dtype = {

                                        'group_memberships_group_id': str,

                                        'group_memberships_user_id': str

                                    })

group_memberships.columns = ['group_id', 'user_id']

group_memberships.describe()
# Take a look at group types



_cat_groups = pd.read_csv('../input/groups.csv', 

                         usecols=['groups_group_type'],

                        squeeze = True).unique()

print(_cat_groups)
cat_groups = CategoricalDtype(categories = _cat_groups, ordered=False)
groups = pd.read_csv('../input/groups.csv',

                         dtype = {

                             'groups_id': str,

                             'groups_group_type': cat_groups

                         })

groups.columns = ['id', 'type']

groups.set_index('id', inplace = True)

groups.sort_index(kind = 'mergesort', inplace = True)

display(groups.head())
matches = pd.read_csv('../input/matches.csv',

                          dtype = {

                              'matches_email_id': str,

                              'matches_question_id': str

                          })

matches.columns = ['email_id', 'question_id']

matches.describe()
# Take a look at professional industry types



_cat_indus = pd.read_csv('../input/professionals.csv', 

                         usecols=['professionals_industry'],

                        squeeze = True).unique()

print(len(_cat_indus))



# It seems like industries are user specified so there is a large amount of variability here. 

# Maybe we can attempt to clean this up and add more structure in a bit...

# For now, we will treat this column as a string
professionals = pd.read_csv('../input/professionals.csv',

                                 infer_datetime_format = True,

                                 dtype = {

                                     'professionals_id': str,

                                     'professionals_location': str,

                                     'professionals_industry': str,

                                     'professionals_headline': str

                                 },

                                parse_dates = ['professionals_date_joined'])

professionals.columns = ['id', 'location', 'industry', 'headline', 'date_joined']

professionals.set_index('id', inplace = True)

professionals.sort_index(kind = 'mergesort', inplace = True)

display(professionals.head())
questions = pd.read_csv('../input/questions.csv',

                       infer_datetime_format = True,

                       dtype = {

                           'questions_id': str,

                           'questions_author_id': str,

                           'questions_title': str,

                           'questions_body': str

                       },

                       parse_dates = ['questions_date_added'])

questions.columns = ['id', 'author_id', 'date_added', 'title', 'body']

questions.set_index('id', inplace = True)

questions.sort_index(kind = 'mergesort', inplace = True)

display(questions.head())
school_memberships = pd.read_csv('../input/school_memberships.csv',

                                dtype = {

                                    'school_memberships_school_id': str,

                                    'school_memberships_user_id': str

                                })

school_memberships.columns = ['school_id', 'user_id']

school_memberships.describe()
students = pd.read_csv('../input/students.csv',

                      infer_datetime_format = True,

                      dtype = {

                          'students_id': str,

                          'students_location': str

                      },

                      parse_dates = ['students_date_joined'])

students.columns = ['id', 'location', 'date_joined']

students.set_index('id', inplace = True)

students.sort_index(kind = 'mergesort', inplace = True)

display(students.head())
questions_tags = pd.read_csv('../input/tag_questions.csv',

                           dtype = {

                               'tag_questions_tag_id': str,

                               'tag_questions_question_id': str

                           })

questions_tags.columns = ['tag_id', 'question_id']

questions_tags.describe()
users_tags = pd.read_csv('../input/tag_users.csv',

                           dtype = {

                               'tag_users_tag_id': str,

                               'tag_users_user_id': str

                           })

users_tags.columns = ['tag_id', 'user_id']

users_tags.describe()
tags = pd.read_csv('../input/tags.csv',

                  dtype = {

                      'tags_tag_id': str,

                      'tags_tag_name': str

                  })

tags.columns = ['id', 'name']

tags.set_index('id', inplace = True)

tags.sort_index(kind = 'mergesort', inplace = True)

display(tags.head())
# Helper functions



# Replace a Series with one where NaN are converted to an empty list

# @args

# s: Series

def conv_nan_list(s):

    s[s.isnull()] = s[s.isnull()].apply(lambda x: [])
# Split group_memberhsips to user -> groups and group -> users



user_groups = pd.DataFrame(

    group_memberships.groupby('user_id')['group_id'].apply(list)

)

user_groups.columns = ['group_ids']

display(user_groups.head())



group_users = pd.DataFrame(

    group_memberships.groupby('group_id')['user_id'].apply(list)

)

group_users.columns = ['user_ids']

display(group_users.head())
# Split matches to email_id -> question_ids and question_id -> email_ids



# HIGH COMPUTE TIME so commented out for now...



# email_questions = matches.groupby('email_id')['question_id'].apply(list)

# display(email_questions.head())



# question_emails = matches.groupby('question_id')['email_id'].apply(list)

# display(question_emails.head())
# Split school_memberships to school_id -> user_ids and user_id -> school_ids



school_users = pd.DataFrame(

    school_memberships.groupby('school_id')['user_id'].apply(list)

)

school_users.columns = ['user_ids']

display(school_users.head())



user_schools = pd.DataFrame(

    school_memberships.groupby('user_id')['school_id'].apply(list)

)

user_schools.columns = ['school_ids']

display(user_schools.head())
# Split questions_tags to question_id -> tag_ids and tag_id -> question_ids



question_tags = pd.DataFrame(

    questions_tags.groupby('question_id')['tag_id'].apply(list)

)

question_tags.columns = ['tag_ids']

display(question_tags.head())



tag_questions = pd.DataFrame(

    questions_tags.groupby('tag_id')['question_id'].apply(list)

)

tag_questions.columns = ['question_ids']

display(tag_questions.head())
# Split users_tags to user_id -> tag_ids and tag_id -> user_ids



user_tags = pd.DataFrame(

    users_tags.groupby('user_id')['tag_id'].apply(list)

)

user_tags.columns = ['tag_ids']

display(user_tags.head())



tag_users = pd.DataFrame(

    users_tags.groupby('tag_id')['user_id'].apply(list)

)

tag_users.columns = ['user_ids']

display(tag_users.head())
# Combine questions + answers + question_tags into a single data frame

qa = questions.join(answers.set_index('question_id'), how = 'left', lsuffix = '_q', rsuffix = '_a')

qa = qa.join(question_tags, how='left')

display(qa.head())
# Combine stsudents + schools + groups + tags into a single data frame



students_full = students.join(user_schools, how='left')

students_full = students_full.join(user_groups, how='left')

students_full = students_full.join(user_tags, how='left')

students_full.columns = ['location', 'date_joined', 'school_ids', 'group_ids', 'tag_ids']



display(students_full.head())
# Combine professionals + schools + groups + tags into a single data frame



professionals_full = professionals.join(user_schools, how='left')

professionals_full = professionals_full.join(user_groups, how='left')

professionals_full = professionals_full.join(user_tags, how='left')

professionals_full.columns = ['location', 'industry', 'headline', 'date_joined',

                             'school_ids', 'group_ids', 'tag_ids']



display(professionals_full.head())
# we will begin by looking at the tags and question_tags dataframes

display(question_tags.head())



tags_list = np.array(tags.index.unique())

display(len(tags_list))

# more than 16k tags!... how many are actually used though? 



_tags_list = questions_tags['tag_id'].unique()

print(len(_tags_list))

# Only 7091 actually used! Let's use these instead...



tags_list = np.array(_tags_list)

tags_list = np.sort(_tags_list, kind='mergesort')

print(tags_list[0:10], tags_list[-10:-1])

# Okay perfect, let's move on...
# How many tags are typically applied to questions?

question_tags['tags_count'] = question_tags['tag_ids'].apply(lambda x: len(x))

display(question_tags.head())



m = np.mean(question_tags['tags_count'])

s = np.std(question_tags['tags_count'])

print(m - s, m, m + s)
# We will then only look at questions that have at least 2 tags (necssary, 1 is not enough) and at most 6 tags

# Let's just see how many don't fit our criteria...

mask = (question_tags['tags_count'] > 2) & (question_tags['tags_count'] < 6)

print(len(question_tags))

question_tags_ok = question_tags.loc[mask]

print(len(question_tags_ok))



# So we have reduced our list from 23k to about 10k or just over a 50% reduction.

display(question_tags_ok.head())
# Some helper functions for this section. We will index our tags and use their indices in our voting matrix

def get_tag_index(tag):

    i = np.searchsorted(tags_list, [str(tag)])[0]

    if i > len(tags_list):

        return None

    return i



def get_tag_indices(tag_list):

    return [get_tag_index(l) for l in tag_list]



# Testing, everything checks out so far!

print(get_tag_index('29'), tags_list[5887])

print(get_tag_indices(['29', '12217']), tags_list[5887], tags_list[429])



question_tags['tag_indices'] = question_tags['tag_ids'].apply(get_tag_indices)

display(question_tags.head())
# Now for the fun stuff! Let's construct our voting matrix

couples = np.array(question_tags['tag_indices'])

display(couples[:10])
N = len(tags_list)

vm = np.zeros((N, N), dtype=int)



# Take the votes

for c in couples:

    for j in c:

        for i in c:

            if i != j:

                vm[i, j] += 1

                

                

print(vm)

print(vm.shape)

print(np.sum(vm[0]))



# How many rows / columns add to 0? Symmetric matrix!

zero_rows = [i for i, row in enumerate(vm) if np.sum(row) == 0]

print(len(zero_rows))



# 343 empty rows! This is due to us not counting EVERY group of tags. There are some unused because they were only used once or in large groups

# In production, we would not do this, but this is just for analysis- even if a tag iss used once we would like to see how it was used

vm = np.delete(vm, zero_rows, axis = 0)

vm = np.delete(vm, zero_rows, axis = 1)



# and make sure to update our tags_list...

tags_list = np.delete(tags_list, zero_rows)

print(len(tags_list))

print(vm.shape)
# Let's see what it looks like to take just the top 5 tags for each category

# A better clustering example would be best here, but we will use this just for now

# top_tags = 