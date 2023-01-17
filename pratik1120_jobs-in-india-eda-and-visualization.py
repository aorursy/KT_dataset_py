import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

from pandas_profiling import ProfileReport

from plotly.offline import iplot

!pip install joypy

import joypy



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")



data = pd.read_csv('../input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
# describing data



data.describe(include='all')
#shape of data



data.shape
drop_var = ['Uniq Id', 'Crawl Timestamp']



data.drop(drop_var, axis=1, inplace=True)
report = ProfileReport(data)
report
pd.options.plotting.backend = "plotly"
print('Missing Values Plot')

plt.figure(figsize=(12,5))

sns.barplot(data=data.isnull().sum().reset_index(), x='index', y=0)

plt.xlabel('Variables')

plt.ylabel('Missing value Count')

plt.show()
# dropping duplicates



data = data.drop_duplicates()
print('How many unique values are present per vairable?')

plt.figure(figsize=(12,5))

sns.barplot(data=data.nunique().reset_index(), x='index', y=0)

plt.xlabel('Variables')

plt.ylabel('Unique value Count')

plt.show()
print("What are the most common job titles?")

df = data['Job Title'].value_counts().reset_index().head(7)

plt.figure(figsize=(10,5))

sns.barplot(data=df, y='index', x='Job Title')

plt.xlabel('')

plt.ylabel('JOB TITLES')

plt.show()
df = data['Job Salary'].dropna().reset_index()

df = df[df['Job Salary'].str.contains('PA')]



print("What are the most common Job Salaries?")

df = df['Job Salary'].value_counts().reset_index().head(10)

plt.figure(figsize=(10,5))

sns.barplot(data=df, y='index', x='Job Salary')

plt.xlabel('')

plt.ylabel('JOB SALARIES')

plt.show()
print('Most common job titles of people working with data')

df = data[data['Job Title'].str.contains('Data')==True]

df = df.dropna().reset_index()

df1 = df['Job Title'].value_counts().reset_index().head()

df1 = df1.sort_values('Job Title', ascending=True)

df1.plot(kind='bar', x='index', y='Job Title', color='Job Title').update_layout(xaxis_title='', yaxis_title='Job Title')
print('Common salary of people working with data')

df = data[data['Job Title'].str.contains('Data')==True]

df = df.dropna().reset_index()

df = df[df['Job Salary'].str.contains('PA')]

df = df['Job Salary'].value_counts().reset_index().head(7)

plt.figure(figsize=(10,5))

sns.barplot(data=df, y='index', x='Job Salary')

plt.xlabel('')

plt.ylabel('DATA JOB SALARIES')

plt.show()
print('What is the most common job experience required?')

df = data['Job Experience Required'].value_counts().reset_index().head(6)

df = df.sort_values('Job Experience Required', ascending=True)

df.plot(kind='bar', x='index', y='Job Experience Required', color='Job Experience Required').update_layout(xaxis_title='Job Experience Required', yaxis_title='')
print('Salary range of Data Scientists: 8-30 LPA')

df = data[data['Job Title']==' Data Scientist']

df1 = df['Job Salary'].value_counts().reset_index()[1:]

df1
print('Salary range of Data Analysts: 5-9 LPA')

df = data[data['Job Title']==' Data Analyst']

df1 = df['Job Salary'].value_counts().reset_index()[1:]

df1
print('Experience required for Data Scientists: 2-9 years')

df = data[data['Job Title']==' Data Scientist']

df1 = df['Job Experience Required'].value_counts().reset_index()[1:]

df1.plot(kind='bar', x='index', y='Job Experience Required', color='Job Experience Required')
print('Experience required for Data Analysts: 5-12 years')

df = data[data['Job Title']==' Data Analyst']

df1 = df['Job Experience Required'].value_counts().reset_index()[1:]

df1.plot(kind='bar', x='index', y='Job Experience Required')
print('Most common location for Data Analyst or Data Scientist jobs')

df = data[data['Job Title'].isin([' Data Scientist',' Data Analyst'])]

df1 = df['Location'].value_counts().reset_index().head()

df1.plot(kind='bar', x='index', y='Location', color='Location')
print('What are the most demanded skills for data scientists?')

df = data[data['Job Title']==' Data Scientist'].reset_index(drop=True)

skills_dict = {}

for i in range(df.shape[0]):

    lst = df['Key Skills'][i].split('|')

    for j in range(len(lst)):

        if lst[j].strip() in skills_dict:

            skills_dict[lst[j].strip()]+=1

        else:

            skills_dict[lst[j].strip()]=1

df1 = pd.DataFrame(skills_dict.items())

df1 = df1.rename(columns={0:"Skill", 1:"NeedCount"})

df1 = df1.sort_values('NeedCount', ascending=True).tail(10)

df1.plot(kind='bar', y='Skill', x='NeedCount', color='NeedCount')
print('What are the most demanded skills for data analysts?')

df = data[data['Job Title']==' Data Analyst'].reset_index(drop=True)

skills_dict = {}

for i in range(df.shape[0]):

    lst = df['Key Skills'][i].split('|')

    for j in range(len(lst)):

        if lst[j].strip() in skills_dict:

            skills_dict[lst[j].strip()]+=1

        else:

            skills_dict[lst[j].strip()]=1

df1 = pd.DataFrame(skills_dict.items())

df1 = df1.rename(columns={0:"Skill", 1:"NeedCount"})

df1 = df1.sort_values('NeedCount', ascending=True).tail(10)

df1.plot(kind='bar', y='Skill', x='NeedCount', color='NeedCount')
print('What roles are data scientists expected to perform?')

df = data[data['Job Title']==' Data Scientist'].reset_index(drop=True)

roles_dict = {}

for i in range(df.shape[0]):

    lst = df['Role'][i].split('/')

    for j in range(len(lst)):

        if lst[j].strip() in roles_dict:

            roles_dict[lst[j].strip()]+=1

        else:

            roles_dict[lst[j].strip()]=1

df1 = pd.DataFrame(roles_dict.items())

df1 = df1.rename(columns={0:"Role", 1:"NeedCount"})

df1 = df1.sort_values('NeedCount', ascending=True).tail(10)

df1.plot(kind='bar', y='Role', x='NeedCount', color='NeedCount')
print('What roles are data analysts expected to perform?')

df = data[data['Job Title']==' Data Analyst'].reset_index(drop=True)

roles_dict = {}

for i in range(df.shape[0]):

    lst = df['Role'][i].split('/')

    for j in range(len(lst)):

        if lst[j].strip() in roles_dict:

            roles_dict[lst[j].strip()]+=1

        else:

            roles_dict[lst[j].strip()]=1

df1 = pd.DataFrame(roles_dict.items())

df1 = df1.rename(columns={0:"Role", 1:"NeedCount"})

df1 = df1.sort_values('NeedCount', ascending=True).tail()

df1.plot(kind='bar', y='Role', x='NeedCount', color='NeedCount')
print('What is the most common functional area for data scientists?')

df = data[data['Job Title']==' Data Scientist'].reset_index(drop=True)

roles_dict = {}

for i in range(df.shape[0]):

    lst = df['Functional Area'][i].split(',')

    for j in range(len(lst)):

        if lst[j].strip() in roles_dict:

            roles_dict[lst[j].strip()]+=1

        else:

            roles_dict[lst[j].strip()]=1

df1 = pd.DataFrame(roles_dict.items())

df1 = df1.rename(columns={0:"Functional Area", 1:"NeedCount"})

df1 = df1.sort_values('NeedCount', ascending=True).tail(10)

df1.plot(kind='bar', y='Functional Area', x='NeedCount', color='NeedCount')
print('What is the most common functional area for data analysts?')

df = data[data['Job Title']==' Data Analyst'].reset_index(drop=True)

roles_dict = {}

for i in range(df.shape[0]):

    lst = df['Functional Area'][i].split(',')

    for j in range(len(lst)):

        if lst[j].strip() in roles_dict:

            roles_dict[lst[j].strip()]+=1

        else:

            roles_dict[lst[j].strip()]=1

df1 = pd.DataFrame(roles_dict.items())

df1 = df1.rename(columns={0:"Functional Area", 1:"NeedCount"})

df1 = df1.sort_values('NeedCount', ascending=True).tail(10)

df1.plot(kind='bar', y='Functional Area', x='NeedCount', color='NeedCount')