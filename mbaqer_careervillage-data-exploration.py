import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from PIL import Image

import wordcloud
students = pd.read_csv('../input/students.csv')

students.head()
print('Latest:',students.students_date_joined.max(), '\n' + 'Earliest:',students.students_date_joined.min())
print('not null: ', students.students_location.notnull().sum(), '\nnull:', students.students_location.isnull().sum())

print (round(students.students_location.isnull().sum()/students.students_location.notnull().sum() *100, 2) , '% of the data is null')
students.students_location.value_counts().head(20)
student_location = students.students_location.value_counts().head(20)

student_location.plot.barh(figsize=(10,10), legend=True)

plt.title('Top Locations of students\n',fontsize='16')

plt.ylabel('Location',fontsize='12')

plt.xlabel('Frequency',fontsize='12')

plt.gca().invert_yaxis()

plt.show()
pros = pd.read_csv("../input/professionals.csv")

pros.head()
print('Latest:',pros.professionals_date_joined.max(), '\n' + 'Earliest:',pros.professionals_date_joined.min())
print('Location: \nnot null: ', pros.professionals_location.notnull().sum(), '\nnull:', pros.professionals_location.isnull().sum())

print (round(pros.professionals_location.isnull().sum()/pros.professionals_location.notnull().sum() *100 , 2), '% of the data is null')

print('Industry: \nnot null: ', pros.professionals_industry.notnull().sum(), '\nnull:', pros.professionals_industry.isnull().sum())

print (round(pros.professionals_industry.isnull().sum()/pros.professionals_industry.notnull().sum() *100, 2) , '% of the data is null')

print('Headline: \nnot null: ', pros.professionals_headline.notnull().sum(), '\nnull:', pros.professionals_headline.isnull().sum())

print (round(pros.professionals_headline.isnull().sum()/pros.professionals_headline.notnull().sum() *100, 2) , '% of the data is null')
pros.professionals_location.value_counts().head(20)
pros_location = pros.professionals_location.value_counts().head(20)

pros_location.plot.barh(figsize=(10,10), legend=True)

plt.title('Top Locations of Professionals\n',fontsize='16')

plt.ylabel('Location',fontsize='12')

plt.xlabel('Frequency',fontsize='12')

plt.gca().invert_yaxis()

plt.show()
pros.professionals_industry.value_counts().head(20)
pros_industry = pros['professionals_industry'].value_counts().head(20)

pros_industry.plot.barh(figsize=(10,10), legend=True)

plt.title('Top Industries of Professionals\n',fontsize='16')

plt.ylabel('Location',fontsize='12')

plt.xlabel('Frequency',fontsize='12')

plt.gca().invert_yaxis()

plt.show()
pros.professionals_headline.value_counts().head(20)
pros_headline = pros.professionals_headline.value_counts().head(20)

pros_headline.plot.barh(figsize=(10,10), legend=True)

plt.title('Top Headlines of Professionals\n',fontsize='16')

plt.ylabel('Location',fontsize='12')

plt.xlabel('Frequency',fontsize='12')

plt.gca().invert_yaxis()

plt.show()
qs = pd.read_csv("DataFiles/questions.csv")

qs.head()
ans = pd.read_csv("DataFiles/answers.csv")

ans.head()
emails = pd.read_csv("DataFiles/emails.csv")

emails.head(0)
matches = pd.read_csv("DataFiles/matches.csv")

matches.head(0)
tags = pd.read_csv("DataFiles/tags.csv")

tags.head(0)
tag_users = pd.read_csv("DataFiles/tag_users.csv")

tag_users.head(0)
tag_qs = pd.read_csv("DataFiles/tag_questions.csv")

tag_qs.head(0)
groups = pd.read_csv("DataFiles/groups.csv")

groups.head(0)
group_m = pd.read_csv("DataFiles/group_memberships.csv")

group_m.head(0)
school = pd.read_csv("DataFiles/school_memberships.csv")

school.head(0)
commnets = pd.read_csv("DataFiles/comments.csv")

commnets.head(0)