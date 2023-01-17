import numpy as np

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

import matplotlib as mpl

import os
print(os.listdir("../input"))
students = pd.read_csv("../input/students.csv", index_col = "students_id", parse_dates = ['students_date_joined'])

professionals = pd.read_csv('../input/professionals.csv', index_col = "professionals_id", parse_dates = ['professionals_date_joined'])

emails = pd.read_csv("../input/emails.csv", index_col = "emails_id")

questions = pd.read_csv('../input/questions.csv', index_col = "questions_id", parse_dates = ['questions_date_added'])

answers = pd.read_csv('../input/answers.csv', index_col = "answers_id", parse_dates = ["answers_date_added"])

tag_questions = pd.read_csv("../input/tag_questions.csv")

tags = pd.read_csv("../input/tags.csv")

matches = pd.read_csv("../input/matches.csv")





students.shape

#students.students_id.nunique()

#students.info()
students.head()
students_locations = students.students_location.value_counts().sort_values(ascending=True).tail(20)

students_locations.plot.barh(figsize=(10, 8), color='b', width=1)

plt.title("Number of students by location", fontsize=20)

plt.xlabel('Number of students', fontsize=12)

plt.show()
print("The number of students without specified location is", students.students_location.isna().sum())
professionals.shape
professionals.head()
prof_locations = professionals.professionals_location.value_counts().sort_values(ascending=True).tail(20)

prof_locations.plot.barh(figsize=(10, 8), color='b', width=1)

plt.title("Number of professionals by location", fontsize=20)

plt.xlabel('Number of professionals', fontsize=12)

plt.show()
print("The number of professionals without specified location is", professionals.professionals_location.isna().sum())
prof_industry = professionals.professionals_industry.value_counts().sort_values(ascending=True).tail(20)

prof_industry.plot.barh(figsize=(10, 8), color='b', width=1)

plt.title("Number of professionals by industry", fontsize=20)

plt.xlabel('Number of professionals', fontsize=12)

plt.show()
print("The number of professionals without specified industry is", professionals.professionals_industry.isna().sum())

print('The number of distinct industries is', professionals.professionals_industry.nunique())
professionals.professionals_headline.sample(20)
questions.shape
questions.head()
print("There are", questions.questions_author_id.nunique(), "unique questions_author_id's, which means that the students who have asked questions asked about 2 questions each on average.")
answers.shape
answers.head()
print("There are", answers.answers_author_id.nunique(), "unique answers_author_id's, which means that the professionals who have given answers have given about 5 answers each on average.")
tag_questions.head()
tags.head()
tag_questions = tag_questions.merge(right=tags, how="left", left_on="tag_questions_tag_id", right_on="tags_tag_id")

tag_questions_groups = tag_questions.tags_tag_name.value_counts().sort_values(ascending=True).tail(20)

tag_questions_groups.plot.barh(figsize=(10, 8), color='b', width=1)

plt.title("Top20 Question tags", fontsize=20)

plt.xlabel('Number of questions with the tag', fontsize=12)

plt.show()
emails.shape
emails.head()
print("There are", emails.emails_recipient_id.nunique(), "unique email recipients.")
emails.emails_frequency_level.replace(["email_notification_daily", "email_notification_immediate", "email_notification_weekly"], ["daily", "immediate", "weekly"], inplace=True)



email_nots = emails.emails_frequency_level.value_counts()



ax = plt.figure()

ax = email_nots.plot.bar(figsize=(10, 8), color='b', width=1, rot = 0)

plt.title("Email notifications", fontsize=20)

ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

ax.tick_params(axis='x', labelsize=14)

plt.show()
matches.shape
matches.rename(columns={'matches_email_id': 'emails_id'}, inplace=True)

#The right join ensures that emails without any questions are also added (one entry with matches_question_id is NA)

matches = pd.merge(matches, emails['emails_frequency_level'].reset_index(), on="emails_id", how="right")

matches.head()
print(matches[matches.emails_frequency_level == "daily"].matches_question_id.count(), "questions were asked in daily emails")

print(matches[matches.emails_frequency_level == "immediate"].matches_question_id.count(), "questions were asked in immediate emails")

print(matches[matches.emails_frequency_level == "weekly"].matches_question_id.count(), "questions were asked in weekly emails")
fig = plt.figure(figsize=(20,15))

grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)



matches_count =  matches.groupby(['emails_id', 'emails_frequency_level']).count()['matches_question_id'].reset_index()

daily_counts = matches_count[matches_count.emails_frequency_level == "daily"].matches_question_id.value_counts().sort_index()

immediate_counts = matches_count[matches_count.emails_frequency_level == "immediate"].matches_question_id.value_counts().sort_index()

weekly_counts = matches_count[matches_count.emails_frequency_level == "weekly"].matches_question_id.value_counts().sort_index()[:31]



ax1 = plt.subplot(grid[0, :1])

ax1 = daily_counts[:4].plot.bar(color='b', width=1, rot=0)

plt.title("Daily emails part 1", fontsize=20)

ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.ylabel('Number of emails', fontsize=14)

plt.xlabel('Number of questions per mail', fontsize=14)



ax11 = plt.subplot(grid[0, 1:])

ax11 = daily_counts[4:31].plot.bar(color='b', width=1, rot=0)

plt.title("Daily emails part 2", fontsize=20)

ax11.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.ylabel('Number of emails', fontsize=14)

plt.xlabel('Number of questions per mail', fontsize=14)



ax2 = plt.subplot(grid[1, :1])

ax2 = immediate_counts.plot.bar(color='b', width=1, rot=0)

plt.title("Immediate emails", fontsize=20)

ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.ylabel('Number of emails', fontsize=14)

plt.xlabel('Number of questions per mail', fontsize=14)



ax3 = plt.subplot(grid[1, 1:])

ax3 = weekly_counts.plot.bar(color='b', width=1, rot=0)

plt.title("Weekly emails", fontsize=20)

ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.ylabel('Number of emails', fontsize=14)

plt.xlabel('Number of questions per mail', fontsize=14)

plt.show()