import pandas as pd

from pandas import Series, DataFrame

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style('white')
stack = pd.read_csv('../input/2016 Stack Overflow Survey Responses.csv')
stack.head()
reputation = stack.rep_range.value_counts()

reputation.plot.bar(figsize=(10,10),width=0.4)

plt.xticks(size = 20)

plt.yticks(size = 20)

plt.ylabel('rep_range',size=20)

plt.xlabel('Range',size=20)

plt.figtext(0.9,0.9,0.9)
visitfrequency = stack.visit_frequency.value_counts()

visitfrequency.plot.bar(figsize=(10,10),width=0.4, stacked=True )

plt.xticks(size = 20)

plt.yticks(size = 20)

plt.ylabel('Range',size=30)

plt.xlabel('Visit_Frequency',size=30)

plt.figtext(0.9,0.9,0.9)

plt.suptitle('Visit Frequency',size=30)

addition = sum(visitfrequency)

percentage = round((visitfrequency * 100) / addition)

percentage
happy_coders=pd.crosstab(stack.job_satisfaction,stack.commit_frequency)

happy_coders["Multiple times a day"].plot.bar(figsize=(10,10),width=0.4)

plt.xticks(size = 10)

plt.yticks(size = 10)

plt.ylabel('Range',size=20)

plt.xlabel('Job Satisfaction',size=20)

# plt.figtext(0.9,0.9,0.9)

plt.suptitle('Happy Coders',size=20)

happy_coders
happy = pd.crosstab(stack.commit_frequency,stack.job_satisfaction)

happy['I love my job'].plot.bar(figsize=(10,10),width=0.4)

plt.xticks(size = 20)

plt.yticks(size = 20)

plt.ylabel('Range',size=20)

plt.xlabel('Job Satisfaction',size=20)

# plt.figtext(0.9,0.9,0.9)

plt.suptitle('Happy Coders',size=30)
checkin_code = stack.commit_frequency.value_counts()

checkin_code.plot.bar(figsize=(10,10),width=0.4)

plt.xticks(size = 20)

plt.yticks(size = 20)

plt.ylabel('Range',size=20)

plt.xlabel('Commit Frequency',size=20)

# plt.figtext(0.9,0.9,0.9)

plt.suptitle('Check-in Code',size=20)

addition = sum(checkin_code)

percentage = round((checkin_code * 100) / addition)

percentage
remote_developers = pd.crosstab(stack.remote,stack.job_satisfaction)

remote_developers['I love my job'].plot.bar(figsize=(10,10),width=0.4)

remote_developer = stack.remote.value_counts()

plt.xticks(size = 20)

plt.yticks(size = 20)

plt.ylabel('Range',size=20)

plt.xlabel('Commit Frequency',size=20)

# plt.figtext(0.9,0.9,0.9)

plt.suptitle('Check-in Code',size=20)

addition = sum(remote_developer)

percentage = round((remote_developer * 100) / addition)

percentage
remote_country = pd.crosstab(stack.remote,stack.country)

individual = remote_country['Argentina'].sum()

percentage = remote_country['Argentina']['Full-time remote']
remote_experienced = pd.crosstab(stack.experience_range,stack.remote)

remote_experienced["Full-time remote"].plot.bar(figsize=(10,10),width=0.4)

plt.xticks(size = 20)

plt.yticks(size = 20)

plt.ylabel('Range',size=20)

plt.xlabel('Experience range',size=20)

# plt.figtext(0.9,0.9,0.9)

plt.suptitle('Remote Developers',size=20)

# addition = sum(remote_experienced)

# percentage = round((remote_experienced * 100) / addition)

# percentage
complete_remote = stack.remote.value_counts()

complete_remote.plot.bar(figsize=(8,8))

plt.xticks(size = 10)

plt.yticks(size = 10)

plt.ylabel('Range',size=10)

plt.xlabel('Experience range',size=10)

# plt.figtext(0.9,0.9,0.9)

plt.suptitle('Remote Developers',size=10)

addition = sum(complete_remote)

percentage = round((complete_remote * 100) / addition)

percentage
stack.women_on_team.value_counts()
company_size = stack.company_size_range.value_counts()

company_size.plot.bar(figsize=(30,30),width=0.6,cmap='terrain')

plt.xticks(size = 70)

plt.yticks(size = 70)

plt.ylabel('Range',size=60)

plt.xlabel('Company-Size',size=60)

# plt.figtext(0.9,0.9,0.9)

plt.suptitle('Company Size',size=70)

addition = sum(company_size)

percentage = round((company_size * 100) / addition)

percentage
industry = stack.industry.value_counts()

industry.plot.bar(figsize=(10,10),width=0.5,cmap='seismic')

plt.xticks(size = 20)

plt.yticks(size = 20)

plt.ylabel('Range',size=20)

plt.xlabel('Industry',size=20)

# plt.figtext(0.9,0.9,0.9)

plt.suptitle('Industry',size=20)

addition = sum(industry)

percentage = round((industry * 100) / addition)

percentage
job_industry = pd.crosstab(stack.industry,stack.job_satisfaction)

job_industry

# .plot.bar(figsize=(30,30),width=0.6)

# plt.xticks(size = 70)

# plt.yticks(size = 70)

# plt.ylabel('Range',size=60)

# plt.xlabel('Company-Size',size=60)

# # plt.figtext(0.9,0.9,0.9)

# plt.suptitle('Company Size',size=70)

# addition = sum(job_industry)

# percentage = round((job_industry * 100) / addition)

# percentage
pd.crosstab(stack.education,stack.salary_range)['More than $200,000'].sort_values().tail()
sns.jointplot(stack['salary_midpoint'],stack['age_midpoint'],kind='hex',color='seagreen')
stack.groupby(['gender','age_range'])[['salary_midpoint']].median()
sns.boxplot(x=stack.age_midpoint, y=stack.salary_midpoint, hue=stack.gender, data=stack, palette="Blues_d")

sns.despine(offset=10, trim=True)