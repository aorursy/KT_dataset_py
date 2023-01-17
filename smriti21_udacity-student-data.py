

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from collections import defaultdict



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


enrollments = pd.read_csv("../input/udacity-enrollments/enrollments.csv")


daily_engagement = pd.read_csv("../input/udacity-daily-engagement/daily_engagement.csv")
submissions =pd.read_csv("/kaggle/input/udacity-project-submissions/project_submissions.csv")
enrollments.head()
enrollments.shape


daily_engagement.shape
enrollments.columns

enrollments.dtypes
enrollments['join_date'] = pd.to_datetime(enrollments['join_date'])
enrollments = enrollments[enrollments['is_udacity']==False]
paid_students = {}

for index,row in enrollments.iterrows():

       if row['days_to_cancel']>7 or row['is_canceled']==False:

            account_key = row['account_key']

            paid_students[account_key]= row['join_date']

            
len(paid_students)
daily_engagement.columns

daily_engagement.dtypes
daily_engagement['utc_date'] = pd.to_datetime(daily_engagement['utc_date'])
paid_engagement= []

for index,row in daily_engagement.iterrows():

    if row['acct'] in paid_students:

        paid_engagement.append(row)

len(paid_engagement)
paid_engagement_for_week = []

for engagement in paid_engagement:

    utc_date = engagement['utc_date']

    join_date = paid_students[engagement['acct']]

    days = (utc_date - join_date).days

    if days < 7 and days>=0:

        paid_engagement_for_week.append(engagement)
total_minutes_spent_in_week = {}

for engagement in paid_engagement_for_week:

    if engagement['acct'] in total_minutes_spent_in_week:

        total_minutes_spent_in_week[engagement['acct']]+=engagement['total_minutes_visited']  

    else:

        total_minutes_spent_in_week[engagement['acct']]=engagement['total_minutes_visited']  

    
max_time_spent = np.max(list(total_minutes_spent_in_week.values()))

max_time_spent

min_time_spent = np.min(list(total_minutes_spent_in_week.values()))

min_time_spent
deviation =  np.std(list(total_minutes_spent_in_week.values()))

deviation
mean_time_spent =  np.mean(list(total_minutes_spent_in_week.values()))

mean_time_spent

lessons_completed = {}

for engagement in paid_engagement_for_week:

    if engagement['acct'] not in lessons_completed:

        lessons_completed[engagement['acct']] = engagement['lessons_completed']

    else:

        lessons_completed[engagement['acct']] += engagement['lessons_completed']

        
maximum = np.max(list(lessons_completed.values()))

maximum
minimum = np.min(list(lessons_completed.values()))

minimum
deviation = np.std(list(lessons_completed.values()))

deviation
mean = np.mean(list(lessons_completed.values()))

mean
courses_visited = {}

for engagement in paid_engagement_for_week:

    if engagement['acct'] not in courses_visited:

        if engagement['num_courses_visited'] > 0:

            courses_visited[engagement['acct']] = 1

    else:

        if engagement['num_courses_visited'] > 0:

            courses_visited[engagement['acct']] += 1 

maximum = np.max(list(courses_visited.values()))

maximum
minimum = np.min(list(courses_visited.values()))

minimum
mean = np.mean(list(courses_visited.values()))

mean
std = np.std(list(courses_visited.values()))

std
submissions.shape
submissions.columns
submissions.dtypes
paid_submissions = []

for index,row in submissions.iterrows():

    if row['account_key'] in paid_students:

        paid_submissions.append(row)

len(paid_submissions)
passed_students = set()

failed_students = set()

for submission in paid_submissions:

    if submission['lesson_key'] == 746169184 or submission['lesson_key'] == 3176718735:

        if submission['assigned_rating'] == 'PASSED' or submission['assigned_rating'] == 'DISTINCTION':

            passed_students.add(submission['account_key'])

        else:

            failed_students.add(submission['account_key'])

        
passing_engagement = []

non_passing_engagement = []

for engagement in paid_engagement_for_week:

    if engagement['acct'] in passed_students:

        passing_engagement.append(engagement)

        p_c+=1

    else:

        non_passing_engagement.append(engagement)

        np_c+=1

print(len(passing_engagement))

print(len(non_passing_engagement))



        
total_minutes_by_passed = []

total_minutes_by_non_passed = []

for row in passing_engagement:

        total_minutes_by_passed.append(row['total_minutes_visited'])

for row in non_passing_engagement:

        total_minutes_by_non_passed.append(row['total_minutes_visited'])

mean_passed = np.sum(total_minutes_by_passed)/len(passed_students)

mean_non_passed = np.sum(total_minutes_by_non_passed)/len(failed_students)

    
print(mean_passed)

print(mean_non_passed)
# use already built course visited dict

courses_by_passed = 0

courses_by_failed = 0

for key,value in courses_visited.items():

    if key in passed_students:

        courses_by_passed+=value

    else:

        courses_by_failed+=value

mean_complete_courses_by_passed = courses_by_passed/len(passed_students)

mean_complete_courses_by_failed = courses_by_failed/len(failed_students)

print(mean_complete_courses_by_passed,mean_complete_courses_by_failed)
len(non_passing_engagement)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.hist(total_minutes_by_passed)

plt.hist(total_minutes_by_non_passed)
courses_by_passed = []

courses_by_failed = []

for key,value in courses_visited.items():

    if key in passed_students:

        courses_by_passed.append(value)

    else:

        courses_by_failed.append(value)

        
import seaborn as sns

plt.hist(courses_by_passed)

plt.hist(courses_by_failed)

lessons_by_passed = []

lessons_by_failed = []

for key,value in lessons_completed.items():

    if key in passed_students:

        lessons_by_passed.append(value)

    else:

        lessons_by_failed.append(value)

        
import seaborn as sns

plt.hist(lessons_by_passed,bins=36)

plt.hist(lessons_by_failed,bins=36)

plt.xlabel('Number of lessons in first week')

plt.ylabel('Students')
