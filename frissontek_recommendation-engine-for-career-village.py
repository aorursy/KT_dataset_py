# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # For Bar Plots

import matplotlib as mpl



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
professionals = pd.read_csv('../input/professionals.csv', parse_dates = ['professionals_date_joined'])

students = pd.read_csv('../input/students.csv', parse_dates = ['students_date_joined'])

groups = pd.read_csv('../input/groups.csv')

group_memberships = pd.read_csv('../input/group_memberships.csv')

emails = pd.read_csv('../input/emails.csv', parse_dates = ['emails_date_sent'])

matches = pd.read_csv('../input/matches.csv')

questions = pd.read_csv('../input/questions.csv', parse_dates = ['questions_date_added'])

answers = pd.read_csv('../input/answers.csv', parse_dates = ['answers_date_added'])

tag_questions = pd.read_csv('../input/tag_questions.csv')

tags = pd.read_csv('../input/tags.csv')



tag_users = pd.read_csv('../input/tag_users.csv')

comments = pd.read_csv('../input/comments.csv')
#Function to show data labels on chart

def add_value_labels(ax, cntin, spacing=1):

    # For each bar: Place a label

    for rect in ax.patches:

        # Get X and Y placement of label from rect.

        y_value = rect.get_height()

        x_value = rect.get_x() + rect.get_width() / 2



        # Number of points between bar and label. Change to your liking.

        space = spacing

        # Vertical alignment for positive values

        va = 'bottom'



        # If value of bar is negative: Place label below bar

        if y_value < 0:

            # Invert space to place label below

            space *= -1

            # Vertically align label at top

            va = 'top'



        # Use Y value as label and format number with one decimal place

        if cntin=='K':

            label = "{:.2f}".format(y_value/1000)+"K"

        else:

            label = "{:.1f}".format(y_value)



        # Create annotation

        ax.annotate(

            label,                      # Use `label` as label

            (x_value, y_value),         # Place label at end of the bar

            xytext=(0, space),          # Vertically shift label by `space`

            textcoords="offset points", # Interpret `xytext` as offset in points

            ha='center',                # Horizontally center label

            va=va)                      # Vertically align label differently for

                                        # positive and negative values.

print("Count of professionals and columns - " + str(professionals.shape))

print("Unique Industries - " + str(professionals.professionals_industry.nunique()))

print("Missing values in Industry - " + str(professionals.professionals_industry.isna().sum()))

print("Unique Locations - " + str(professionals.professionals_location.nunique()))

print("Missing values in Location - " + str(professionals.professionals_location.isna().sum()))
professionals_industries = professionals.professionals_industry.value_counts().sort_values(ascending=True).tail(14)

ax = professionals_industries.plot(kind='barh',figsize=(10, 8),width=0.8) 

ax.set_title("Top 14 industries Professionals belong to", fontsize=20)

ax.set_xlabel('Number of Professionals', fontsize=12)

for p in ax.patches:

     ax.annotate(str(p.get_width()), (p.get_width() * 1.005, p.get_y() * 1.005))
professionals_locations = professionals.professionals_location.value_counts().sort_values(ascending=True).tail(14)

ax = professionals_locations.plot(kind='barh',figsize=(10, 8),width=0.8) 

ax.set_title("Top 14 Locations Professionals hail from", fontsize=20)

ax.set_xlabel('Number of Professionals', fontsize=12)

for p in ax.patches:

     ax.annotate(str(p.get_width()), (p.get_width() * 1.005, p.get_y() * 1.005))
import datetime

df_profs = professionals.copy()

df_profs['YearJoined']=df_profs['professionals_date_joined'].dt.year

prof_yrjoined = df_profs.groupby('YearJoined').count()

prof_yrjoined = prof_yrjoined.drop ('professionals_id',axis=1)

prof_yrjoined = prof_yrjoined.drop ('professionals_location',axis=1)

prof_yrjoined = prof_yrjoined.drop ('professionals_industry',axis=1)

prof_yrjoined = prof_yrjoined.drop ('professionals_headline',axis=1)

prof_yrjoined = prof_yrjoined.rename(columns={'professionals_date_joined':'Count'})



prof_yrjoined.head()
plt.plot(prof_yrjoined, color='orange')

plt.xlabel('Year Joined')

plt.ylabel('Number of Professionals')

plt.title('Trend of number of Professionals joining Career Village')

plt.show()
import datetime

df_profs = professionals.copy()

df_profs['YearJoined']=df_profs['professionals_date_joined'].dt.year

prof_yrjoined = df_profs[df_profs['YearJoined'].isin(['2016','2017','2018'])][['YearJoined','professionals_industry','professionals_date_joined']].groupby(['YearJoined','professionals_industry']).count()



prof_yrjoined=prof_yrjoined[prof_yrjoined['professionals_date_joined'] > 109]

prof_yrjoined = prof_yrjoined.rename(columns={'professionals_date_joined':'Count'})

df_profscopy = prof_yrjoined.unstack()

df_profscopy.plot(kind='barh', stacked=True, figsize=[16,6])
import datetime

df_profs = professionals.copy()

df_profs['YearJoined']=df_profs['professionals_date_joined'].dt.year

df_profs['Count'] = df_profs[df_profs['YearJoined'].isin(['2016','2017','2018'])][['YearJoined','professionals_industry','professionals_date_joined']].groupby(['YearJoined','professionals_industry']).transform('count')



prof_yrjoined=df_profs[df_profs['Count'] > 109][['YearJoined','professionals_industry','Count']].drop_duplicates()



group_size=prof_yrjoined.groupby('YearJoined').count()['Count'].tolist()

group_names=prof_yrjoined['YearJoined'].unique().tolist()

subgroup_names=prof_yrjoined['professionals_industry'].tolist()

subgroup_size=prof_yrjoined['Count'].tolist()

# Libraries

import matplotlib.pyplot as plt

 

# Make data: I have 3 groups and 7 subgroups

 

# Create colors

a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

 

# First Ring (outside)

fig, ax = plt.subplots()

ax.axis('equal')

mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[a(0.6), b(0.6), c(0.6)] )

plt.setp( mypie, width=0.3, edgecolor='white')

 

# Second Ring (Inside)

mypie2, _ = ax.pie(subgroup_size, radius=3.5-0.3, labels=subgroup_names, labeldistance=0.8, colors=[a(0.5), a(0.4), a(0.3), b(0.5), b(0.4), c(0.6), c(0.5), c(0.4), c(0.3), c(0.2)])

plt.setp( mypie2, width=0.4, edgecolor='white')

plt.margins(0,0)

 

# show it

plt.show()
print("Count of Students and columns - " + str(students.shape))

print("Unique Locations - " + str(students.students_location.nunique()))

print("Missing values in Location - " + str(students.students_location.isna().sum()))
import datetime

df_students = students.copy()

df_students['YearJoined']=df_students['students_date_joined'].dt.year

student_yrjoined = df_students.groupby('YearJoined').count()

student_yrjoined = student_yrjoined.drop ('students_id',axis=1)

student_yrjoined = student_yrjoined.drop ('students_location',axis=1)

student_yrjoined = student_yrjoined.rename(columns={'students_date_joined':'Count'})



student_yrjoined.head()
plt.plot(student_yrjoined, color='orange')

plt.xlabel('Year Joined')

plt.ylabel('Number of Students')

plt.title('Trend of number of Students joining Career Village')

plt.show()
students_locations = students.students_location.value_counts().sort_values(ascending=True).tail(14)

ax = students_locations.plot(kind='barh',figsize=(10, 8),width=0.8) 

ax.set_title("Top 14 Locations Students hail from", fontsize=20)

ax.set_xlabel('Number of Professionals', fontsize=12)

for p in ax.patches:

     ax.annotate(str(p.get_width()), (p.get_width() * 1.005, p.get_y() * 1.005))
group_det = group_memberships.join(groups, how = 'inner').groupby('groups_group_type').count()

group_det = group_det.drop ('group_memberships_user_id',axis=1)

group_det = group_det.drop ('groups_id',axis=1)

group_det = group_det.rename(columns={'group_memberships_group_id':'Count'})

group_det
prof_grp = professionals.merge(group_memberships, how = 'left',

                                            left_on ='professionals_id',

                                            right_on ='group_memberships_user_id')



prof_grp = pd.merge(groups, prof_grp, how='inner',

                left_on ='groups_id',

                right_on ='group_memberships_group_id')



prof_grp=prof_grp.groupby('groups_group_type').count()



prof_grp = prof_grp.rename(columns={'group_memberships_group_id':'Professionals_Count'})

prof_grp[['Professionals_Count']]



St_grp = students.merge(group_memberships, how = 'left',

                                            left_on ='students_id',

                                            right_on ='group_memberships_user_id')



St_grp = pd.merge(groups, St_grp, how='inner',

                left_on ='groups_id',

                right_on ='group_memberships_group_id')



St_grp=St_grp.groupby('groups_group_type').count()



St_grp = St_grp.rename(columns={'group_memberships_group_id':'Students_Count'})

St_grp[['Students_Count']]

allingrp=prof_grp[['Professionals_Count']].join(St_grp[['Students_Count']], how='inner')

allingrp
atag = pd.merge(answers, tag_questions, how = 'inner',

                                            left_on ='answers_question_id',

                                            right_on ='tag_questions_question_id')

print('Number of total tags are ' + str(len(atag)) + ' for ' + str(len(atag.answers_id.unique())) + ' questions')
qtag = pd.merge(pd.merge(questions, tag_questions, how='inner',

                                           left_on = 'questions_id',

                                           right_on = 'tag_questions_question_id'),

                                                tags, how='inner',

                                                left_on='tag_questions_tag_id',

                                                right_on='tags_tag_id')

ptag = pd.merge(pd.merge(professionals, tag_users, how='inner',

                                           left_on = 'professionals_id',

                                           right_on = 'tag_users_user_id'),

                                                tags, how='inner',

                                                left_on='tag_users_tag_id',

                                                right_on='tags_tag_id')

qtag=qtag.groupby('tags_tag_name').count()

ptag=ptag.groupby('tags_tag_name').count()

qtag = qtag.rename(columns={'tag_questions_question_id':'TagsInQuestions'})

ptag = ptag.rename(columns={'tag_users_user_id':'TagsFollowedByProfessionals'})

tagcounts=qtag[['TagsInQuestions']].join(ptag[['TagsFollowedByProfessionals']], how='inner')

tagcounts=tagcounts[(tagcounts['TagsInQuestions'] > 100) & (tagcounts['TagsFollowedByProfessionals'] > 100)]

tagcounts=tagcounts.sort_values(by=['TagsInQuestions'], ascending=False)

taglist=tagcounts.index[:].tolist()
import numpy as np

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

from matplotlib import rc

import pandas as pd

 

# y-axis in bold

rc('font', size='10')

 

# Values of each group

bars1 = tagcounts['TagsInQuestions'].tolist()

bars2 = tagcounts['TagsFollowedByProfessionals'].tolist()

 

# Heights of bars1 + bars2

bars = np.add(bars1, bars2).tolist()

 

# The position of the bars on the x-axis

r = list(range(0,len(tagcounts)))

 

# Names of group and bar width

names= taglist

barWidth = 0.8

 

plt.figure(figsize=(20, 8))  # width:20, height:8

# Create brown bars

plt.bar(r, bars1, color='#ff9700', edgecolor='white', align='edge', width=barWidth)

# Create green bars (middle), on top of the firs ones

plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white',align='edge', width=barWidth)



b1 = mpatches.Patch(facecolor='#ff9700', label='In Questions', linewidth = 0.5, edgecolor = 'black')

b2 = mpatches.Patch(facecolor='#557f2d', label = 'Followed by Professionals', linewidth = 0.5, edgecolor = 'black')

plt.legend(handles=[b1, b2], title="Tags", loc=1, fontsize='12', fancybox=True)



# Custom X axis

plt.xticks(r, names)

plt.xticks(rotation=90)

plt.xlabel("Tags", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("Popular Tags within questions vs tags followed by professionals",fontsize=20)



# Show graphic

plt.show()
questions_received = matches.merge(right=emails, how = 'left',

                                            left_on ='matches_email_id',

                                            right_on ='emails_id')

emailsreceived = professionals.merge(right=emails, how = 'left',

                                            left_on ='professionals_id',

                                            right_on ='emails_recipient_id')

answersgiven_cnt = answers.groupby(['answers_author_id']).count()

answersgiven_cnt = answersgiven_cnt.sort_values('answers_author_id')

answersgiven_cnt = answersgiven_cnt.reset_index()

answersgiven_cnt = answersgiven_cnt.rename(columns={'answers_id': 'answers_given'})

answersgiven_cnt = answersgiven_cnt.drop(['answers_date_added','answers_body','answers_question_id'], axis=1)



questions_received_cnt = questions_received.groupby(['emails_recipient_id']).count()

questions_received_cnt = questions_received_cnt.sort_values('emails_recipient_id')

questions_received_cnt = questions_received_cnt.reset_index()

questions_received_cnt = questions_received_cnt.rename(columns={'emails_id': 'questions_received'})

questions_received_cnt = questions_received_cnt.drop(['matches_email_id','matches_question_id','emails_date_sent','emails_frequency_level'], axis=1)



emailsreceived_cnt = emailsreceived.groupby(['emails_recipient_id','professionals_date_joined','professionals_location','professionals_industry']).count()

emailsreceived_cnt = emailsreceived_cnt.sort_values('emails_recipient_id')

emailsreceived_cnt = emailsreceived_cnt.reset_index()

emailsreceived_cnt = emailsreceived_cnt.rename(columns={'emails_id': 'emails_received'})

emailsreceived_cnt = emailsreceived_cnt.drop(['professionals_id','professionals_headline','emails_date_sent','emails_frequency_level'], axis=1)



prof_e_q_det = emailsreceived_cnt.merge(questions_received_cnt, how='inner')

prof_e_q_det = prof_e_q_det.merge(answersgiven_cnt, how='left',

                                 left_on ='emails_recipient_id',

                                 right_on ='answers_author_id')

prof_e_q_det = prof_e_q_det.drop(['answers_author_id'],axis=1)

prof_e_q_det = prof_e_q_det.fillna(0)

prof_e_q_det.head()

plt.figure(figsize=(10,10))

plt.scatter(prof_e_q_det['questions_received'],prof_e_q_det['answers_given'],  color='k', s=25, alpha=0.2)

plt.xlim(-5, 90)

plt.ylim(-5,50)

plt.plot([-5,90], [-5,50], 'k-', color = 'r')



plt.xlabel('Questions Received')

plt.ylabel('Answers Given')

plt.title('Questions Received vs Answers Given by Professionals')

plt.legend()

plt.show()
answers_tags = answers.merge(right=tag_questions, how = 'inner',

                                            left_on ='answers_question_id',

                                            right_on ='tag_questions_question_id')

answers_tags = answers_tags.merge(right=tag_users, how = 'left',

                                            left_on =['tag_questions_tag_id','answers_author_id'],

                                            right_on =['tag_users_tag_id','tag_users_user_id'])



question_tags_followed = answers_tags.fillna(-1).groupby(['tag_questions_tag_id','tag_users_tag_id']).count()

question_tags_followed = question_tags_followed.sort_values('tag_questions_tag_id')

question_tags_followed = question_tags_followed.reset_index()

question_tags_followed = question_tags_followed.rename(columns={'answers_id': 'Count'})

question_tags_followed = question_tags_followed.merge(right=tags, how = 'inner',

                                            left_on ='tag_questions_tag_id',

                                            right_on ='tags_tag_id')

question_tags_followed = question_tags_followed.drop(['answers_author_id','answers_question_id','answers_date_added','answers_body','tag_questions_question_id','tag_users_user_id','tags_tag_id'], axis=1)

question_tags_followed.head(10)







followed = question_tags_followed[question_tags_followed['tag_users_tag_id']>0]

notfollowed = question_tags_followed[question_tags_followed['tag_users_tag_id']<0]

foldf = pd.merge(followed, notfollowed, how='outer',

                                left_on='tag_questions_tag_id',

                                right_on='tag_questions_tag_id')

foldf['diff']=(foldf['Count_x']-foldf['Count_y']).abs()

foldf = foldf.sort_values('diff', ascending=False)

foldf=foldf.head(40)
import numpy as np

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

from matplotlib import rc

import pandas as pd

 

# y-axis in bold

rc('font', size='10')

 

# Values of each group

bars1 = foldf['Count_x']

bars2 = foldf['Count_y']





# Heights of bars1 + bars2

bars = np.add(bars1, bars2).tolist()

 

# The position of the bars on the x-axis

r = list(range(0,40))

 

# Names of group and bar width

names= foldf['tags_tag_name_x']

barWidth = 0.8

 

plt.figure(figsize=(20, 8))  # width:20, height:8

# Create brown bars

plt.bar(r, bars1, color='#ff9700', edgecolor='white', align='edge', width=barWidth)

# Create green bars (middle), on top of the firs ones

plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white',align='edge', width=barWidth)



b1 = mpatches.Patch(facecolor='#ff9700', label='Followed', linewidth = 0.5, edgecolor = 'black')

b2 = mpatches.Patch(facecolor='#557f2d', label = 'Not Followed', linewidth = 0.5, edgecolor = 'black')

plt.legend(handles=[b1, b2], title="Question Tags", loc=1, fontsize='12', fancybox=True)



# Custom X axis

plt.xticks(r, names)

plt.xticks(rotation=90)

plt.xlabel("Tags", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("Questions answered by professionals who followed tags or not",fontsize=20)



# Show graphic

plt.show()
followedtags=followed.copy()

followedtags = followedtags[['tags_tag_name','Count']].sort_values('Count',ascending=True).tail(25)

followedtags.set_index('tags_tag_name', inplace=True)



ax = followedtags.plot(kind='barh',figsize=(12, 8),width=0.8) 

ax.set_title("Top 25 tags followed by Professionals who answered", fontsize=20)

ax.set_xlabel('Number of Professionals', fontsize=12)

ax.set_ylabel('Tag Names', fontsize=12)

for p in ax.patches:

     ax.annotate(str(p.get_width()), (p.get_width() * 1.005, p.get_y() * 1.005))
notfollowedtags=answers_tags.copy()

notfollowedtags=notfollowedtags[notfollowedtags.tag_users_user_id.isnull()==True]

notfollowedtags = notfollowedtags.groupby(['answers_author_id','answers_id']).count()

notfollowedtags = notfollowedtags.reset_index()



notfollowedtags = notfollowedtags.groupby(['answers_author_id']).count()

notfollowedtags = notfollowedtags.rename(columns={'answers_id': 'Count'})

notfollowedtags = notfollowedtags.merge(right=professionals, how = 'inner',

                                            left_on ='answers_author_id',

                                            right_on ='professionals_id')

notfollowedtags = notfollowedtags.drop(['tag_questions_tag_id','tag_users_tag_id','answers_question_id','answers_date_added','answers_body','tag_questions_question_id','tag_users_user_id','professionals_headline'], axis=1)



notfollowedtags.head()
notfollowedtags_industry=notfollowedtags.copy()

notfollowedtags_industry = notfollowedtags_industry.groupby(['professionals_industry'])['Count'].sum()

notfollowedtags_industry = notfollowedtags_industry.reset_index()

notfollowedtags_industry = notfollowedtags_industry.sort_values('Count',ascending=True).tail(25)

notfollowedtags_industry.set_index('professionals_industry', inplace=True)

ax = notfollowedtags_industry.plot(kind='barh',figsize=(12, 8),width=0.8) 

ax.set_title("Top 25 industries of professionals who answered but did not follow tags", fontsize=20)

ax.set_xlabel('Number of Professionals', fontsize=12)

ax.set_ylabel('Industries', fontsize=12)

for p in ax.patches:

     ax.annotate(str(p.get_width()), (p.get_width() * 1.005, p.get_y() * 1.005))
notfollowedtags_location=notfollowedtags.copy()

notfollowedtags_location = notfollowedtags_location.groupby(['professionals_location'])['Count'].sum()

notfollowedtags_location = notfollowedtags_location.reset_index()

notfollowedtags_location = notfollowedtags_location.sort_values('Count',ascending=True).tail(25)

notfollowedtags_location.set_index('professionals_location', inplace=True)

ax = notfollowedtags_location.plot(kind='barh',figsize=(12, 8),width=0.8) 

ax.set_title("Top 25 Locations of professionals who answered but did not follow tags", fontsize=20)

ax.set_xlabel('Number of Professionals', fontsize=12)

ax.set_ylabel('Industries', fontsize=12)

for p in ax.patches:

     ax.annotate(str(p.get_width()), (p.get_width() * 1.005, p.get_y() * 1.005))
print("Count of Questions and columns - " + str(questions.shape))

print("Mail notifications sent for above questions - " + str(emails.shape))

print("Questions mailed in above mails - " + str(matches.shape))

print("Number of Questions for which mail notification sent - " + str(matches.matches_question_id.nunique()))
emails_matches = pd.merge(emails, matches, how='inner',

                               left_on='emails_id',

                               right_on='matches_email_id')

emails_response = pd.merge(emails_matches, answers, how='left',

                               left_on=['matches_question_id','emails_recipient_id'],

                               right_on=['answers_question_id','answers_author_id'])

emails_noresponse = emails_response[emails_response['answers_question_id'].isnull()]

emails_noresponse = emails_noresponse.groupby(['emails_frequency_level']).count()

emails_noresponse = emails_noresponse.rename(columns={'matches_question_id': 'Count'})

emails_noresponse = emails_noresponse.reset_index()



emails_freq_level_cnt = emails_matches.copy()

emails_freq_level_cnt = emails_freq_level_cnt.groupby(['emails_frequency_level']).count()

emails_freq_level_cnt = emails_freq_level_cnt.rename(columns={'matches_email_id': 'Count'})

emails_freq_level_cnt = emails_freq_level_cnt.drop(['emails_id','emails_recipient_id','emails_date_sent','matches_question_id'],axis=1)

emails_freq_level_cnt = emails_freq_level_cnt.reset_index()



plt.figure(figsize=(20,12))

ax1=plt.subplot(221)

emails_freq_level_cnt[['emails_frequency_level','Count']].plot(kind='bar', ax=ax1, legend=False, width=0.3)



ax1.set_title("Total Questions sent per Email Frequency level", fontsize=15)

ax1.set_xticklabels(emails_freq_level_cnt['emails_frequency_level'], fontsize=10, rotation=30)

add_value_labels(ax1,'N')



ax2=plt.subplot(222)

emails_noresponse[['emails_frequency_level','Count']].plot(kind='bar', ax=ax2, legend=False, width=0.3)



ax2.set_title("Response not received per Email Frequency level", fontsize=15)

ax2.set_xticklabels(emails_noresponse['emails_frequency_level'], fontsize=10, rotation=30)

add_value_labels(ax2,'N')



import matplotlib.pyplot as plt

from pandas.plotting import table

plt.figure(figsize=(12,6))

# plot chart

ax1 = plt.subplot(121, aspect='equal')

emails_freq_level_cnt.plot(kind='pie', y = 'Count', ax=ax1, autopct='%1.2f%%', 

 startangle=90, shadow=False, labels=emails_freq_level_cnt['emails_frequency_level'], legend = False, fontsize=11)



# plot table

ax2 = plt.subplot(122)

plt.axis('off')

tbl = table(ax2, emails_freq_level_cnt, loc='center')

tbl.auto_set_font_size(False)

tbl.set_fontsize(9)

plt.show()
yearmonthwise_emails=emails_matches.copy()

yearmonthwise_emails['YearMailed']=yearmonthwise_emails['emails_date_sent'].dt.year

yearmonthwise_emails['MonthMailed']=yearmonthwise_emails['emails_date_sent'].dt.month



plt.figure(figsize=(20,12))

ax1=plt.subplot(221)

yearmonthwise_emails[['YearMailed','emails_date_sent']].groupby('YearMailed').count().sort_values('YearMailed').plot(kind='bar', ax=ax1, legend=False)

ylabels = ['{:,.1f}'.format(y) + 'M' for y in ax1.get_yticks()/1000000]

ax1.set_yticklabels(ylabels)

add_value_labels(ax1,'K')



ax2=plt.subplot(222)

yearmonthwise_emails[['MonthMailed','emails_date_sent']].groupby('MonthMailed').count().sort_values('emails_date_sent', ascending=False).plot(kind='bar', ax=ax2, legend=False)

ylabels = ['{:,.1f}'.format(y) + 'M' for y in ax2.get_yticks()/1000000]

ax2.set_yticklabels(ylabels)

add_value_labels(ax2,'K',spacing=2)
answers_emails_response = emails_response[emails_response['answers_question_id'].isnull()==False]

answers_noemails_response=answers[~answers[['answers_question_id','answers_author_id']].apply(tuple,1).isin(emails_response[['matches_question_id','emails_recipient_id']].apply(tuple,1))]

print("Count of Answers and columns - " + str(answers.shape))

mcnt=emails_response[emails_response['answers_question_id'].isnull()==False].count()['emails_recipient_id']

print("Answers in response of Mail notifications - " + str(mcnt))

print("Number of Professionals who answered without Mail notifications - " + str(emails_response[emails_response['answers_question_id'].isnull()].emails_recipient_id.nunique()))

print("Number of Answers without Mail notifications - " + str(answers_noemails_response.count()['answers_id']))
dfProf = emails.groupby('emails_recipient_id').first().reset_index()

answers_noemails_response_prof=answers_noemails_response.copy()

answers_noemails_response_prof = pd.merge(answers_noemails_response, dfProf, how = 'left',

                                    left_on='answers_author_id',

                                    right_on='emails_recipient_id')



answers_noemails_response_prof.fillna('No_Prior_Email_Sent', inplace=True)



plt.figure(figsize=(20,12))

ax1=plt.subplot(221)

answers_emails_response[['emails_frequency_level','answers_question_id']].groupby('emails_frequency_level').count().plot(kind='bar', ax=ax1, legend=False, width=0.3)

add_value_labels(ax1,'N')

ax1.set_title("Answered questions in response to mail", fontsize=20)

ax1.set_xlabel('')



ax2=plt.subplot(222)

answers_noemails_response_prof[['emails_frequency_level','answers_id']].groupby('emails_frequency_level').count().plot(kind='bar', ax=ax2, legend=False, width=0.3)

ax2.set_xlabel('')

ax2.set_title("Answered questions without mail notification", fontsize=20)

add_value_labels(ax2,'N')

questions_sent_mean_yearly = emails_matches.copy()

questions_sent_mean_yearly['YearSent']=questions_sent_mean_yearly['emails_date_sent'].dt.year

questions_sent_mean_yearly = questions_sent_mean_yearly.groupby(['YearSent','matches_question_id']).count().reset_index().groupby('YearSent')['emails_id'].mean()



answers_mean_yearly = answers.copy()

answers_mean_yearly = answers_mean_yearly.merge(questions, how='inner',

                                               left_on='answers_question_id',

                                               right_on='questions_id')

answers_mean_yearly['YearAsked']=answers_mean_yearly['questions_date_added'].dt.year

answers_mean_yearly = answers_mean_yearly.groupby(['YearAsked','answers_question_id']).count().reset_index().groupby('YearAsked')['questions_id'].mean()



plt.figure(figsize=(20,12))

ax1=plt.subplot(221)

ax1 = questions_sent_mean_yearly.plot(kind='barh',width=0.5) 

ax1.set_title("Average number of mails sent per question year wise", fontsize=15)

for p in ax1.patches:

     ax1.annotate(str("{:.2f}".format(p.get_width())), (p.get_width() * 1.005, p.get_y() * 1.005))



ax2=plt.subplot(222)

ax2 = answers_mean_yearly.plot(kind='barh',width=0.5) 

ax2.set_title("Average number of answers per question year wise", fontsize=15)

for p in ax2.patches:

     ax2.annotate(str("{:.2f}".format(p.get_width())), (p.get_width() * 1.005, p.get_y() * 1.005))
questions_det_cnt = questions.copy()

questions_det_cnt['YearAsked']=questions_det_cnt['questions_date_added'].dt.year

questions_yrly_cnt = questions_det_cnt.groupby('YearAsked').count().reset_index()

questions_yrly_cnt = questions_yrly_cnt.rename(columns={'questions_id': 'questions_asked'})

questions_yrly_cnt = questions_yrly_cnt.drop(['questions_author_id','questions_date_added','questions_title','questions_body'],axis=1)



questions_yrly_cnt_notanswered = questions_det_cnt[~questions_det_cnt.questions_id.isin(answers.answers_question_id)][['YearAsked','questions_id']].groupby('YearAsked').count().reset_index()

questions_yrly_cnt_notanswered = questions_yrly_cnt_notanswered.rename(columns={'questions_id': 'not_answered'})



questions_yrly_cnt = questions_yrly_cnt.merge(questions_yrly_cnt_notanswered, how='left',

                                           left_on='YearAsked',

                                           right_on='YearAsked')



questions_yrly_cnt.fillna(0, inplace=True)





questions_yrly_cnt['percent_notanswered']= 100 * (questions_yrly_cnt['not_answered'] / questions_yrly_cnt['questions_asked'])

questions_yrly_cnt.head(10)



questions_yrly_cnt_chart = questions_yrly_cnt.copy()

questions_yrly_cnt_chart = questions_yrly_cnt_chart.drop(['percent_notanswered'],axis=1)

plt.figure(figsize=(20,12))

ax1=plt.subplot(221)

questions_yrly_cnt_chart.groupby('YearAsked').sum().plot(kind='bar', ax=ax1, legend=True, width=0.5)



ax1.set_title("Questions Asked vs Questions Unanswered year wise", fontsize=20)

ax1.set_xlabel('')

add_value_labels(ax1,'N')





questions_yrly_cnt_chart=questions_yrly_cnt_chart[questions_yrly_cnt['not_answered']>0]



colors = ['yellowgreen','red','violet','lightskyblue','white','lightcoral']

x=questions_yrly_cnt_chart['YearAsked']

y=questions_yrly_cnt_chart['not_answered']

porcent = 100.*y/y.sum()



ax2=plt.subplot(222)

ax2, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]



plt.legend(ax2, labels, loc='upper right', bbox_to_anchor=(-0.1, 1.),

           fontsize=8)

plt.title("% Distribution of questions unanswered till date", fontsize=20)

sort_legend = True

if sort_legend:

    ax2, labels, dummy =  zip(*sorted(zip(ax2, labels, y),

                                          key=lambda x: x[2],

                                          reverse=True))
speedtoanswer = answers.copy()

#speedtoanswer['YearAnswered']=speedtoanswer['answers_date_added'].dt.year

speedtoanswer=speedtoanswer.sort_values(['answers_question_id','answers_date_added'], ascending=True).groupby('answers_question_id').first()

speedtoanswer=speedtoanswer.merge(questions, how='inner',

                                     left_on='answers_question_id',

                                     right_on='questions_id')

speedtoanswer['first_answer_days']=speedtoanswer['answers_date_added']-speedtoanswer['questions_date_added']

speedtoanswer['first_answer_days']=speedtoanswer['first_answer_days']/ np.timedelta64(1, 'D')

speedtoanswer['YearAsked']=speedtoanswer['questions_date_added'].dt.year

speedtoanswer=speedtoanswer.groupby('YearAsked')['first_answer_days'].mean()

speedtoanswer.head(10)
plt.plot(speedtoanswer, color='orange')

plt.xlabel('Year Joined')

plt.ylabel('Average speed')

plt.title('Trend of average days to get first answer for the questions asked')

plt.show()
activitylevel_prof = answers.copy()

activitylevel_prof = activitylevel_prof.merge(professionals, how='inner',

                                             left_on='answers_author_id',

                                             right_on='professionals_id')

activitylevel_prof['ProfJoiningYear']=activitylevel_prof['professionals_date_joined'].dt.year

activitylevel_prof['TimeToAnswer']=(activitylevel_prof['answers_date_added']-activitylevel_prof['professionals_date_joined']).abs()

activitylevel_prof['TimeToAnswer']=activitylevel_prof['TimeToAnswer']/ np.timedelta64(1, 'D')

activitylevel_prof['WeekofAnswer']=round(activitylevel_prof['TimeToAnswer']/7,0)

activitylevel_prof=activitylevel_prof.groupby(pd.cut(activitylevel_prof['WeekofAnswer'],np.arange(0, 400, 25))).count()['answers_id']



ax = activitylevel_prof.plot.bar(rot=0, color="b", figsize=(20,6))

add_value_labels(ax,'N')

plt.xlabel('Week after joining')

plt.ylabel('Number of questions answered')

plt.title('Trend of number of questions answered by professionals since joining')

plt.show()

ans_mailsent=emails_response[emails_response['answers_question_id'].isnull()==False][['answers_id','answers_question_id','answers_author_id','answers_date_added']]

ans_nomailsent=answers_noemails_response.copy()

ans_nomailsent=answers_noemails_response[['answers_id','answers_question_id','answers_author_id','answers_date_added']]

q_ans = questions.copy()

q_ans['YearAsked']=q_ans['questions_date_added'].dt.year

ans_mailsent=ans_mailsent.merge(q_ans, how='inner',

                               left_on='answers_question_id',

                               right_on='questions_id')

ans_mailsent['DaysToAnswer_MailSent']=ans_mailsent['answers_date_added']-ans_mailsent['questions_date_added']

ans_mailsent['DaysToAnswer_MailSent'] = ans_mailsent['DaysToAnswer_MailSent'] / np.timedelta64(1, 'D')

ans_mailsent=ans_mailsent.groupby('YearAsked')['DaysToAnswer_MailSent'].mean().reset_index()



ans_nomailsent=ans_nomailsent.merge(q_ans, how='inner',

                               left_on='answers_question_id',

                               right_on='questions_id')

ans_nomailsent['DaysToAnswer_NoMailSent']=ans_nomailsent['answers_date_added']-ans_nomailsent['questions_date_added']

ans_nomailsent['DaysToAnswer_NoMailSent'] = ans_nomailsent['DaysToAnswer_NoMailSent'] / np.timedelta64(1, 'D')

ans_nomailsent=ans_nomailsent.groupby('YearAsked')['DaysToAnswer_NoMailSent'].mean().reset_index()



ans_avgspeedtoanswer = ans_nomailsent.merge(ans_mailsent, how='outer',

                                           left_on='YearAsked',

                                           right_on='YearAsked')

ans_avgspeedtoanswer.set_index('YearAsked')
ans_avgspeedtoanswer.plot(x='YearAsked', y=['DaysToAnswer_NoMailSent', 'DaysToAnswer_MailSent'], color=['green','blue'])

plt.xlabel('Year When Question Added')

plt.ylabel('Average speed to Answer')

plt.title('Trend of average days taken to answer when mail sent and without mail notification')

plt.show()
answers_prof_student_loc = answers.copy()

answers_prof_student_loc = answers_prof_student_loc.merge(professionals, how='inner',

                                                         left_on='answers_author_id',

                                                         right_on='professionals_id')

answers_prof_student_loc = answers_prof_student_loc.merge(questions, how='inner',

                                                         left_on='answers_question_id',

                                                         right_on='questions_id')

answers_prof_student_loc = answers_prof_student_loc.merge(students, how='inner',

                                                         left_on='questions_author_id',

                                                         right_on='students_id')

answers_prof_student_loc = answers_prof_student_loc[['professionals_location', 'professionals_industry','students_location']]



# group by industry and professionals location to get number of unique locations of professionals

answers_prof_ind_loc = answers_prof_student_loc.groupby(['professionals_industry','professionals_location']).count().reset_index()

answers_prof_ind_loc = answers_prof_ind_loc.groupby(['professionals_industry']).count()['professionals_location'].reset_index()

answers_prof_ind_loc = answers_prof_ind_loc.rename(columns={'professionals_location': 'ProfessionalCount'})



# group by industry and students location to get number of unique locations of students

answers_stud_ind_loc = answers_prof_student_loc.groupby(['professionals_industry','students_location']).count().reset_index()

answers_stud_ind_loc = answers_stud_ind_loc.groupby(['professionals_industry']).count()['students_location'].reset_index()

answers_stud_ind_loc = answers_stud_ind_loc.rename(columns={'students_location': 'StudentCount'})



industrywise_locations = answers_stud_ind_loc.merge(answers_prof_ind_loc, how='outer',

                                                   left_on='professionals_industry',

                                                   right_on='professionals_industry')

industrywise_locations=industrywise_locations.sort_values('ProfessionalCount', ascending=False).head(50)

industrywise_locations.head(10)
import numpy as np

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

from matplotlib import rc

import pandas as pd

 

# y-axis in bold

rc('font', size='10')

 

# Values of each group

bars1 = industrywise_locations['ProfessionalCount']

bars2 = industrywise_locations['StudentCount']





# Heights of bars1 + bars2

bars = np.add(bars1, bars2).tolist()

 

# The position of the bars on the x-axis

r = list(range(0,50))

 

# Names of group and bar width

names= industrywise_locations['professionals_industry']

barWidth = 0.8

 

plt.figure(figsize=(20, 8))  # width:20, height:8

# Create brown bars

plt.bar(r, bars1, color='#ff9700', edgecolor='white', align='edge', width=barWidth)

# Create green bars (middle), on top of the firs ones

plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white',align='edge', width=barWidth)



b1 = mpatches.Patch(facecolor='#ff9700', label='Unique Professional Locations', linewidth = 0.5, edgecolor = 'black')

b2 = mpatches.Patch(facecolor='#557f2d', label = 'Unique Student Locations', linewidth = 0.5, edgecolor = 'black')

plt.legend(handles=[b1, b2], title="Question Tags", loc=1, fontsize='12', fancybox=True)



# Custom X axis

plt.xticks(r, names)

plt.xticks(rotation=90)

plt.xlabel("Industry", fontsize=18)

plt.ylabel("Unique Locations", fontsize=18)

plt.title("Number of Unique locations of professionals and students industry wise",fontsize=20)



# Show graphic

plt.show()
#remove_punctuation

def replacepuntuation(s):

    import string

    for c in string.punctuation:

        s=s.replace(c," ")

    return s
#Function to create bag of words for professionals to be able to recommend relevant questions

def create_profinfo_textlib(q_date):

    professionals_info = tag_users.merge(right=tags, how = 'left',

                                            left_on ='tag_users_tag_id',

                                            right_on ='tags_tag_id')



    professionals_info=professionals_info.drop(['tags_tag_id','tag_users_tag_id'],axis=1)

    professionals_info =professionals_info.pivot_table(index='tag_users_user_id',values='tags_tag_name',aggfunc=lambda x: " ".join(x))

    professionals_info = professionals.merge(professionals_info, how='left',

                                            left_on='professionals_id',

                                            right_on='tag_users_user_id')

    professionals_info = professionals_info[professionals_info['professionals_date_joined']<=q_date]

    professionals_info=professionals_info.fillna('')

    professionals_info['bow']= professionals_info['professionals_headline'] + ' ' + professionals_info['tags_tag_name']

    professionals_info['ibow']=professionals_info['professionals_industry']

    professionals_info=professionals_info.drop(['professionals_location','professionals_industry', 'professionals_headline', 'professionals_date_joined', 'tags_tag_name'],axis=1)

    professionals_answers=professionals_info.merge(answers, how='left',

                                            left_on='professionals_id',

                                            right_on='answers_author_id')

    professionals_answers=professionals_answers.merge(questions, how='left',

                                            left_on='answers_question_id',

                                            right_on='questions_id')



    professionals_answers_tags=professionals_answers.merge(tag_questions, how='left',

                                            left_on='answers_question_id',

                                            right_on='tag_questions_question_id')





    professionals_answers_tags = professionals_answers_tags.merge(right=tags, how = 'left',

                                            left_on ='tag_questions_tag_id',

                                            right_on ='tags_tag_id')

    professionals_answers_tags=professionals_answers_tags.drop(['bow','answers_id','tag_questions_tag_id','tag_questions_question_id','tags_tag_id','answers_author_id','answers_question_id','answers_date_added','answers_body','questions_id','questions_author_id','questions_date_added','questions_title','questions_body'],axis=1)

    professionals_answers_tags=professionals_answers_tags.fillna('')

    professionals_answers_tags =professionals_answers_tags.pivot_table(index='professionals_id',values='tags_tag_name',aggfunc=lambda x: " ".join(x))



    professionals_answers=professionals_answers.fillna('')

    professionals_answers['qbow']= professionals_answers['questions_title'] + ' ' + professionals_answers['questions_body']

    professionals_answers=professionals_answers.drop(['bow','answers_id','answers_author_id','answers_question_id','answers_date_added','answers_body','questions_id','questions_author_id','questions_date_added','questions_title','questions_body'],axis=1)

    professionals_answers =professionals_answers.pivot_table(index='professionals_id',values='qbow',aggfunc=lambda x: " ".join(x))

    professionals_info=professionals_info.merge(professionals_answers, how='left',

                                            left_on='professionals_id',

                                            right_on='professionals_id')

    professionals_info=professionals_info.merge(professionals_answers_tags, how='left',

                                            left_on='professionals_id',

                                            right_on='professionals_id')

    

    professionals_info['qbow']=professionals_info['qbow'] + ' ' +  professionals_info['tags_tag_name']

    professionals_info=professionals_info.drop(['tags_tag_name'], axis=1)

    return professionals_info
#Function to create bag of words for questions to find similar questions asked by other students

def create_quesinfo_textlib(q_date):

    question_info = questions.copy()

    question_info['bow']=question_info['questions_title'] + ' ' + question_info['questions_body']

    question_info = question_info[question_info['questions_date_added']<=q_date]

    question_tags = question_info.merge(right=tag_questions, how = 'left',

                                            left_on ='questions_id',

                                            right_on ='tag_questions_question_id')



    question_tags = question_tags.merge(right=tags, how = 'left',

                                            left_on ='tag_questions_tag_id',

                                            right_on ='tags_tag_id')



    question_tags=question_tags.drop(['questions_author_id','questions_date_added', 'questions_title','questions_body','bow','tag_questions_question_id','tags_tag_id'],axis=1)

    question_tags=question_tags.fillna('')

    question_tags =question_tags.pivot_table(index='questions_id',values='tags_tag_name',aggfunc=lambda x: " ".join(x))

    question_tags.head()



    question_info = question_info.merge(question_tags, how='left',

                                            left_on='questions_id',

                                            right_on='questions_id')

    

    question_info = question_info[['questions_id','bow','tags_tag_name']]

    return question_info
# This is the start point when any question is asked in system. Pass questions_id as parameter to my recommendation engine

# Below are the different question ids I have tested my recommender. 

q_id='2f6a9a99d9b24e5baa50d40d0ba50a75'

q_id='eb0027b3dcd04d88b76a493fc1558c15'

q_id='4c6d71aaf2724b9f8d439ae086d4f3da'

q_id='caca9ab7e13d4297a82b9abe8f11f0b8'

q_id='eb80205482e4424cad8f16bc25aa2d9c'

q_id='baa937b4cd184a22acfd76249d25042c'

q_id='6351c23f2e144b359c3301d40b3d81ef'

q_id='c9bd1bedf1e341799026ea304dec6e3c'



#Find the question date based on which we will create dictionary of questions and professional details

quest = questions[questions['questions_id']==q_id]

q_date=quest['questions_date_added'].values[0]



#Create bag of words for finding similar questions

question_info=create_quesinfo_textlib(q_date)



#Create bag of words to find professionals whom to recommend this question

professionals_info=create_profinfo_textlib(q_date)
# Using TF-IDF model we will compare bag of words of question asked with professionals which existed when question was asked

#Below we are creating questions dictionary which existed when question was asked

import gensim

import nltk



from nltk.tokenize import word_tokenize



print("Number of Questions:",len(question_info['bow']))

print("Tokenizing data...")

gen_docs = [[replacepuntuation(w.lower()) for w in word_tokenize(text)] 

            for text in question_info['bow']]

print("Creating dictionary...")

dictionary = gensim.corpora.Dictionary(gen_docs)

print("Creating Document-Term Matrix...")

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

print("Creating TF-IDF Model...")

tf_idf = gensim.models.TfidfModel(corpus)

print("Creating Similarity Checker...")



sims = gensim.similarities.Similarity("",tf_idf[corpus],num_features=len(dictionary))

print("Processing Completed!")
import warnings

warnings.filterwarnings("ignore")

# compare bag of words of questions asked and other questions dictionary

question_asked = question_info[question_info['questions_id']==q_id]

Query=question_asked['bow'].values[0]

qtags=question_asked['tags_tag_name'].values[0]



query_doc = [replacepuntuation(w.lower()) for w in word_tokenize(Query)]

query_doc_bow = dictionary.doc2bow(query_doc)

query_doc_tf_idf = tf_idf[query_doc_bow]

sim_threshold=0.40

sim=sims[query_doc_tf_idf]

question_info['sim']=sim

similar_questions=question_info[(question_info['sim']>=sim_threshold) & (question_info['sim'] < 1)]

similar_questions=similar_questions.sort_values('sim',ascending=False)

top3_simq = similar_questions

top3_simq_ans = top3_simq[top3_simq['questions_id'].isin(answers['answers_question_id'])]

top3_simq_ans= top3_simq_ans.merge(answers, how='left',

                                  left_on='questions_id',

                                  right_on='answers_question_id')

top3_simq_ans= top3_simq_ans.merge(professionals,how='left',

                                  left_on='answers_author_id',

                                  right_on='professionals_id')



#Use sklearn to find the closest match to the industry when we find similar questions 

import nltk, string

from sklearn.feature_extraction.text import TfidfVectorizer



remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)



def normalize(text):

    return nltk.word_tokenize(text.lower().translate(remove_punctuation_map))



vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')



def cosine_sim(text1, text2):

    tfidf = vectorizer.fit_transform([text1, text2])

    return ((tfidf * tfidf.T).A)[0,1]



#Extract bag of words of similar questions and industry names of professionals who answered those questions

industries= top3_simq_ans[top3_simq_ans['professionals_industry'].isnull()==False][['professionals_industry','bow']].values.tolist()

#industries= top3_simq_ans[top3_simq_ans['professionals_industry'].isnull()==False][['professionals_industry']].values.flatten()

top3_simq = top3_simq_ans[top3_simq_ans['professionals_industry'].isnull()==False][['bow']].values.flatten()

p_rank = pd.DataFrame( columns=['industry','bow', 'cos'])

i = 0

while i < len(industries):

    cos = cosine_sim(qtags, industries[i][0])

    p_rank.loc[i] = industries[i][0],industries[i][1],cos

    i += 1



#Filter industry names who are closest match

industry_list=np.unique(p_rank[p_rank['cos']>0].values[:,0]).flatten()

#top3_simq=np.unique(p_rank[p_rank['cos']>0].values[:,1]).flatten()



#Extract question tags of question asked

questions_bow1=qtags

#Extract question title and body of asked question as well as similar questions as found by TF-IDF similarity checker

questions_bow2=Query + ' ' + ' '.join(top3_simq)

#EXtract the relevant industry names

questions_bow3=' '.join(industry_list)

print(questions_bow1)                        

print(questions_bow2)                        

print(questions_bow3)  
# Using TF-IDF model we will compare bag of words of question with professionals bag of words 

#which existed when question was asked

#Below we are creating professionals dictionary for professionals which existed in system when question was asked

#3 comparisions will be done to get relevant professional matches

#This is the first dictionary of professionals which will contain their details like headline and tags they follow

import gensim

import nltk



from nltk.tokenize import word_tokenize



print("Number of Professionals:",len(professionals_info['bow']))

print("Tokenizing data...")

gen_docs_prof = [[replacepuntuation(w.lower()) for w in word_tokenize(text)] 

            for text in professionals_info['bow']]

print("Creating dictionary...")

dictionary_prof = gensim.corpora.Dictionary(gen_docs_prof)

print("Creating Document-Term Matrix...")

corpus_prof = [dictionary_prof.doc2bow(gen_doc) for gen_doc in gen_docs_prof]

print("Creating TF-IDF Model...")

tf_idf_prof = gensim.models.TfidfModel(corpus_prof)

print("Creating Similarity Checker...")

sims_prof = gensim.similarities.Similarity("",tf_idf_prof[corpus_prof],num_features=len(dictionary_prof))

print("Processing Completed!")
#3 comparisions will be done to get relevant professional matches

#This is the second dictionary of professionals which will contain their details like questions they answered earlier

import gensim

import nltk



from nltk.tokenize import word_tokenize



print("Number of Professionals:",len(professionals_info['qbow']))

print("Tokenizing data for professionals answers...")

gen_docs_prof_q = [[replacepuntuation(w.lower()) for w in word_tokenize(text)] 

            for text in professionals_info['qbow']]

print("Creating dictionary...")

dictionary_prof_q = gensim.corpora.Dictionary(gen_docs_prof_q)

print("Creating Document-Term Matrix...")

corpus_prof_q = [dictionary_prof_q.doc2bow(gen_doc) for gen_doc in gen_docs_prof_q]

print("Creating TF-IDF Model...")

tf_idf_prof_q = gensim.models.TfidfModel(corpus_prof_q)

print("Creating Similarity Checker...")

sims_prof_q = gensim.similarities.Similarity("",tf_idf_prof_q[corpus_prof_q],num_features=len(dictionary_prof_q))

print("Processing Completed!")
#3 comparisions will be done to get relevant professional matches

#This is the third dictionary of professionals which will contain their details like Industry they belong to

import gensim

import nltk



from nltk.tokenize import word_tokenize



print("Number of Professionals:",len(professionals_info['ibow']))

print("Tokenizing data for professionals industry...")

gen_docs_prof_i = [[replacepuntuation(w.lower()) for w in word_tokenize(text)] 

            for text in professionals_info['ibow']]

print("Creating dictionary...")

dictionary_prof_i = gensim.corpora.Dictionary(gen_docs_prof_i)

print("Creating Document-Term Matrix...")

corpus_prof_i = [dictionary_prof_i.doc2bow(gen_doc) for gen_doc in gen_docs_prof_i]

print("Creating TF-IDF Model...")

tf_idf_prof_i = gensim.models.TfidfModel(corpus_prof_i)

print("Creating Similarity Checker...")

sims_prof_i = gensim.similarities.Similarity("",tf_idf_prof_i[corpus_prof_i],num_features=len(dictionary_prof_i))

print("Processing Completed!")
import warnings

warnings.filterwarnings("ignore")

#3 comparisions will be done to get relevant professional matches

#we saw dictionary of professionals which will contain their details. Each one will be compared with similar dictionary

#for questions we seen above. I have given weights and thresholds for the similarity match of all 3 comparisions

#depending on the importance of the data based on analysis done



query_doc1 = [replacepuntuation(w.lower()) for w in word_tokenize(questions_bow1)]

query_doc2 = [replacepuntuation(w.lower()) for w in word_tokenize(questions_bow2)]

query_doc3 = [replacepuntuation(w.lower()) for w in word_tokenize(questions_bow3)]



query_doc_bow = dictionary_prof.doc2bow(query_doc1)

query_doc_tf_idf = tf_idf_prof[query_doc_bow]

sim1_threshold=0.1

sim1=sims_prof[query_doc_tf_idf]

professionals_info['sim1']=sim1



query_doc_bow_q= dictionary_prof_q.doc2bow(query_doc2)

query_doc_tf_idf_q = tf_idf_prof_q[query_doc_bow_q]

sim2_threshold=0.3

sim2=sims_prof_q[query_doc_tf_idf_q]

professionals_info['sim2']=sim2



query_doc_bow_i= dictionary_prof_i.doc2bow(query_doc3)

query_doc_tf_idf_i = tf_idf_prof_i[query_doc_bow_i]

sim3_threshold=0.3

sim3=sims_prof_i[query_doc_tf_idf_i]

professionals_info['sim3']=sim3



professionals_info['sim'] = professionals_info['sim1'] + professionals_info['sim2'] + + professionals_info['sim3']

professionals_info['sim12'] = professionals_info['sim1'] + professionals_info['sim2']

professionals_info['sim13'] = professionals_info['sim1'] + professionals_info['sim3']

professionals_info['sim23'] = professionals_info['sim2'] + professionals_info['sim3']

sim12_threshold=0.4

sim13_threshold=0.3

sim23_threshold=0.3



suggested_professionals=professionals_info[(((professionals_info['sim2']>=sim2_threshold) & (professionals_info['sim13']>0)) | \

                                            (professionals_info['sim1']>=sim13_threshold) | \

                                            (professionals_info['sim2']>=sim12_threshold) | \

                                            (((professionals_info['sim2']==0) & (professionals_info['sim12']>=sim12_threshold)) & \

                                            ((professionals_info['sim13']>=sim13_threshold) | (professionals_info['sim1']>=sim1_threshold)) & \

                                             ((professionals_info['sim23']>=sim23_threshold) | (professionals_info['sim3']>=sim3_threshold))))]



#Here I have just added up the 3 match resuts. We can set weight to all the 3 results and sort based on sum in order

#to tweak to get best results

suggested_professionals=suggested_professionals.sort_values('sim',ascending=False)



#This engine will pull 125 relevant matches considering that not many mails should be sent per question

#If needed the number can be altered or below clause can be removed

if round(len(suggested_professionals)) > 125:

    display_size=125

else:

    display_size=round(len(suggested_professionals))

    

top3_simq = suggested_professionals.head(display_size)

#This will give relevant professionals to recommend questions to

top3_simq
#This is a validation summary to list details of old recommender which career village uses vs new recommender created above

validateres = top3_simq.copy()

ares=answers[answers['answers_question_id']==q_id]

print('question id ' +q_id)

pres=validateres[validateres['professionals_id'].isin(ares['answers_author_id'])]

nres=validateres[~validateres['professionals_id'].isin(ares['answers_author_id'])]

mres = matches[matches['matches_question_id']==q_id]

emails_response_res=emails_response[emails_response['matches_question_id']==q_id]

pores=ares[ares[['answers_question_id','answers_author_id']].apply(tuple,1).isin(emails_response_res[['matches_question_id','emails_recipient_id']].apply(tuple,1))]

new_matches=validateres[~validateres['professionals_id'].isin(emails_response_res['emails_recipient_id'])]

mcnt=mres.count().values[0]

pocnt=pores.count().values[0]

new_matches_cnt=new_matches.count().values[0]

nocnt=mcnt-pocnt

acnt=ares.count().values[0]

rcnt=top3_simq.count().values[0]

pcnt=pres.count().values[0]

ncnt=nres.count().values[0]

# as per the email sent and details stored in matches file

print('total mails sent by old recommender: ' + str(mcnt))

# as per the answers received in system

print('total answers: ' + str(acnt))

# Did any professional who was mailed answered the question

print('positive: ' + str(pocnt))

# How many professionals did not respond to this question even after getting mail

print('negative: ' + str(nocnt))

# As per the relevant matches displayed above by my recommender

print('total mails sent by my recommender: ' + str(rcnt))

oldcnt=top3_simq[top3_simq['professionals_id'].isin(answers['answers_author_id'])].count().values[0]

newcnt=top3_simq[~top3_simq['professionals_id'].isin(answers['answers_author_id'])].count().values[0]

# How many professionals have been at some point active in system by answering questions

print('active professionals ' + str(oldcnt))

# How many professionals are recommended this question who have not answered any question or are new to career village

print('Inactive professionals ' + str(newcnt))

# How many professionals are recommended by above recommender who have actually answered this question.

# This is because we are passing questions_id which was asked earlier and validating success of this recommender

print('positive: ' + str(pcnt))

# How many professionals are recommended this question and did not answer. Though we have not actually compared results

# of old recommender with new, we do not know how many new professionals are recommended and may be probable to answer if

# they get mail

print('negative: ' + str(ncnt))

print('New matches added: ' + str(new_matches_cnt))
# This is validation result of why people who have answered were not recommended by new recommender

# Mainly I have noticed that people who answered are not recommended by my recommender is because professionals have joined

# later than the date when question asked

import numpy as np

notincl=ares[~ares['answers_author_id'].isin(pres['professionals_id'])]

profdet=notincl.merge(professionals, how='inner',

                  left_on='answers_author_id',

                  right_on='professionals_id')

profdet['q_date']=q_date

profdet['comments']=''

profdet['comments']=np.where((profdet['q_date'] < profdet['professionals_date_joined']), \

                             profdet['comments'] + ' ' + 'Professional not joined when question asked', \

                             profdet['comments'])



profdet[['q_date','professionals_date_joined','answers_date_added','comments']]
# this is just a place to know bag of words of any professional in the result set of recommender

# It will show all the 3 bag of words of professionals which is used to compare with question bag of words

p_id='fa15dfc3aa1744919d070ced9bd3fe98'

print(top3_simq[top3_simq['professionals_id']==p_id]['bow'].values[0])

print('\n\n'+top3_simq[top3_simq['professionals_id']==p_id]['qbow'].values[0])

print('\n\n'+top3_simq[top3_simq['professionals_id']==p_id]['ibow'].values[0])