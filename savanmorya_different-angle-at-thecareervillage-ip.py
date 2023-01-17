#importing workers

import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import datetime as DT #for today's date

import wordcloud

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
!ls ../input
data_folder = "../input/data-science-for-good-careervillage"
!ls $data_folder
#Reading data

professionals = pd.read_csv(os.path.join(data_folder, 'professionals.csv'))

tag_users = pd.read_csv(os.path.join(data_folder, 'tag_users.csv'))

students = pd.read_csv(os.path.join(data_folder, 'students.csv'))

tag_questions = pd.read_csv(os.path.join(data_folder, 'tag_questions.csv'))

groups = pd.read_csv(os.path.join(data_folder, 'groups.csv'))

emails = pd.read_csv(os.path.join(data_folder, 'emails.csv'))

group_memberships = pd.read_csv(os.path.join(data_folder, 'group_memberships.csv'))

answers = pd.read_csv(os.path.join(data_folder, 'answers.csv'))

comments = pd.read_csv(os.path.join(data_folder, 'comments.csv'))

matches = pd.read_csv(os.path.join(data_folder, 'matches.csv'))

tags = pd.read_csv(os.path.join(data_folder, 'tags.csv'))

questions = pd.read_csv(os.path.join(data_folder, 'questions.csv'))

school_memberships = pd.read_csv(os.path.join(data_folder, 'school_memberships.csv'))
plt.figure(figsize=(15, 15))

plt.imshow(plt.imread('../input/er-diagram/erd.png'), interpolation='bilinear', aspect='auto')

plt.axis("off")

plt.show()
professionals.head()
professionals = professionals.replace(np.nan, '', regex=True)
#timestamp to year converter

def tm_stamp_year(data_joined):

    now = pd.Timestamp(DT.datetime.now())

    time_diff = now.tz_localize('UTC').tz_convert('Asia/Kolkata') - pd.Timestamp(data_joined).tz_convert('Asia/Kolkata')

    return round(time_diff.components.days/365,1)
#professionals exeperience

professionals['Exeperience(year)'] = professionals['professionals_date_joined'].apply(tm_stamp_year)
professionals.head()
plt.hist(professionals['Exeperience(year)'], normed=True, bins=8)

plt.xlabel('Exeperience')
#focus on highly exeperienced professionals

plt.figure(figsize=(20, 10))

plt.imshow(

    wordcloud.WordCloud(

        min_font_size=6,

        background_color='white',

        width=4000,

        height=2000

    ).generate(' '.join(professionals[professionals['Exeperience(year)'] > 6]['professionals_location'].values)),

    interpolation='bilinear'

)

plt.axis("off")

plt.show()
students.head()
students = students.replace(np.nan, '', regex=True)
#Students exeperience

students['Exeperience(year)'] = students['students_date_joined'].apply(tm_stamp_year)
students.head()
plt.hist(students['Exeperience(year)'], normed=True, bins=8)

plt.xlabel('Exeperience')
#focus on highly exeperienced professionals

plt.figure(figsize=(20, 10))

plt.imshow(

    wordcloud.WordCloud(

        min_font_size=6,

        background_color='white',

        width=4000,

        height=2000

    ).generate(' '.join(students[students['Exeperience(year)'] > 7]['students_location'].values)),

    interpolation='bilinear'

)

plt.axis("off")

plt.show()