import pandas as pd

import numpy as np

import os
data_folder = "../input"

answer_scores = pd.read_csv(os.path.join(data_folder, 'answer_scores.csv'))

answers = pd.read_csv(os.path.join(data_folder, 'answers.csv'))

comments = pd.read_csv(os.path.join(data_folder, 'comments.csv'))

emails = pd.read_csv(os.path.join(data_folder, 'emails.csv'))

group_memberships = pd.read_csv(os.path.join(data_folder, 'group_memberships.csv'))

groups = pd.read_csv(os.path.join(data_folder, 'groups.csv'))

matches = pd.read_csv(os.path.join(data_folder, 'matches.csv'))

professionals = pd.read_csv(os.path.join(data_folder, 'professionals.csv'))

question_scores = pd.read_csv(os.path.join(data_folder, 'question_scores.csv'))

questions = pd.read_csv(os.path.join(data_folder, 'questions.csv'))

school_memberships = pd.read_csv(os.path.join(data_folder, 'school_memberships.csv'))

students = pd.read_csv(os.path.join(data_folder, 'students.csv'))

tag_questions = pd.read_csv(os.path.join(data_folder, 'tag_questions.csv'))

tag_users = pd.read_csv(os.path.join(data_folder, 'tag_users.csv'))

tags = pd.read_csv(os.path.join(data_folder, 'tags.csv'))
# "Hearts" scores for each answer.



answer_scores.info()

answer_scores.head(3)
# Answers are what this is all about! Answers get posted in response to questions. Answers can only be posted by users 

# who are registered as Professionals. However, if someone has changed their registration type after joining, they may 

# show up as the author of an Answer even if they are no longer a Professional.



answers.info()

answers.head(3)
# answers.answers_date_added.apply(type).head(3)

# len(answers[answers['answers_date_added'].str.contains('UTC\+0000')])
"""

 51107 rows matched based on answer id

"""

    

ai = np.array(answers['answers_id'])

i = np.array(answer_scores['id'])

cnt = 0

for k in ai:

    if k in i:

        cnt += 1

cnt
cnt
answer_fixed = answers.merge(answer_scores, how='outer', left_on='answers_id', right_on='id')

len(answer_fixed)

answer_fixed.info()

answer_fixed.head(3)
"""

    answer scores can be ranged from 0 to 30. The higher the score is, more people heart this answer. 

"""



answer_fixed.groupby('score').count()
# Comments can be made on Answers or Questions. We refer to whichever the comment is posted to as the "parent" of that 

# comment. Comments can be posted by any type of user. Our favorite comments tend to have "Thank you" in them :)



comments.info()

comments.head(3)
"""

    84.04% of comments belongs to answers

"""

# a1 = np.array(answers['answers_id'])

# c = np.array(comments['comments_parent_content_id'])

# cnt = 0

# for k in c:

#     if k in a1:

#         cnt += 1

# cnt / len(c)  



"""

    30.31% of comments include Thank you 

"""

len(comments[comments['comments_body'].str.contains('Thank you', na=False)]) / len(comments['comments_body'])
# Each email corresponds to one specific email to one specific recipient. The frequency_level refers to the type of 

# email template which includes immediate emails sent right after a question is asked, daily digests, and weekly 

# digests.



emails.info()

emails.head(3)
"""

    Emails sent to recipients are based on 3 types(descending order): Daily, Immediate and Weekly.

"""



emails.groupby('emails_frequency_level').count()
"""

    How many emails did one recipient receive under current situation. 

    One recipient received number of emails ranges from 1 to 3496. 

    Total recipients are around 22,168. 

"""



get_frequency_emails = emails.groupby('emails_recipient_id')['emails_id'].count()

get_frequency_emails.to_frame().sort_values(by=['emails_id'])
# Any type of user can join any group. There are only a handful of groups so far.



group_memberships.info()

group_memberships.head(3)
group_count = group_memberships.groupby('group_memberships_group_id').count()

"""

    46 unique groups.

    One group can have members ranged from 1 to 117. 

"""

# len(group_count) 

group_count.sort_values('group_memberships_user_id')
group_membership_count = group_memberships.groupby('group_memberships_user_id').count()

"""

    727 unique members. 

    A member joined 1 to 14 groups. 

"""

len(group_membership_count)

group_membership_count.sort_values('group_memberships_group_id')
# Each group has a "type". For privacy reasons we have to leave the group names off.



groups.info()

groups.head(3)
"""

    Total 7 unique group types.

    Youth Program has the most groups, and Club/Competition/Interest Group have the least group. 

"""



groups.groupby('groups_group_type').count().sort_values(by='groups_id')
groups_fixed = groups.merge(group_memberships, how='right', left_on='groups_id', 

                            right_on='group_memberships_group_id')

# groups_fixed[groups_fixed['groups_id']==groups_fixed['group_memberships_group_id']].count()

del groups_fixed['group_memberships_group_id']

groups_fixed
# Each row tells you which questions were included in emails. If an email contains only one question, that email's 

# ID will show up here only once. If an email contains 10 questions, that email's ID would show up here 10 times.



matches.info()

matches.head(3)
"""

    One email can contain questions from 1 to 268. 

"""

matches.groupby('matches_email_id').count().sort_values(by='matches_question_id')

# matches.loc[matches['matches_email_id']==569938]
# We call our volunteers "Professionals", but we might as well call them Superheroes. They're the grown ups who 

# volunteer their time to answer questions on the site.



professionals.info()

professionals.head(3)
"""

    Professionals(Superheroes/volunteers) locate in 2582 areas. 

    New York has the most professionals. 

"""



professionals.groupby('professionals_location').count().sort_values(by='professionals_id')
"""

    Professionals can be from 2470 industries. 

"""



professionals.groupby('professionals_industry').count().sort_values(by='professionals_id')
"""

    There are 22272 titles among professionals.

"""



professionals.groupby('professionals_headline').count().sort_values(by='professionals_id')
# "Hearts" scores for each question.



question_scores.info()

question_scores.head(3)
"""

    Questions can be heart from 0 to more than 100. 

    It means how popular current question is. 

"""



question_scores.groupby('score').count().sort_values(by='id')
# Questions get posted by students. Sometimes they're very advanced. Sometimes they're just getting started. It's 

# all fair game, as long as it's relevant to the student's future professional success.



questions.info()

questions.head(3)
questions_fixed = questions.merge(question_scores, how='outer', 

                                  left_on='questions_id', right_on='id')

del questions_fixed['id']

questions_fixed.sort_values(by='score',ascending=False)
# Just like group_memberships, but for schools instead.



school_memberships.info()

school_memberships.head(3)
"""

    School has number of users from 1 to 45. 

"""



school_memberships.groupby('school_memberships_school_id').count().sort_values(by='school_memberships_user_id')
"""

    One user can have more than one schools? 

"""



school_memberships.groupby('school_memberships_user_id').count().sort_values(by='school_memberships_school_id')
# Students are the most important people on CareerVillage.org. They tend to range in age from about 14 to 24. 

# They're all over the world, and they're the reason we exist!



students.info()

students.head(3)
"""

    Students can locate in 5480 areas. 

    New York has the most student accounts. 

"""



students.groupby('students_location').count().sort_values(by='students_id')
# Every question can be hashtagged. We track the hashtag-to-question pairings, and put them into this file.



tag_questions.info()

tag_questions.head(3)
"""

    Tags can have 1 to 3744 questions. 

"""



tag_questions.groupby('tag_questions_tag_id').count().sort_values(by='tag_questions_question_id')
"""

    One question has up to 54 tags. 

"""



tag_questions.groupby('tag_questions_question_id').count().sort_values(by='tag_questions_tag_id')
# Users of any type can follow a hashtag. This shows you which hashtags each user follows.



tag_users.info()

tag_users.head(3)
"""

    One tag can be followed by up to 3135 users. 

"""



tag_users.groupby('tag_users_tag_id').count().sort_values(by='tag_users_user_id')
"""

    One user follows up to 82 tags. 

"""



tag_users.groupby('tag_users_user_id').count().sort_values(by='tag_users_tag_id')
# Each tag gets a name.



tags.info()

tags.head(3)