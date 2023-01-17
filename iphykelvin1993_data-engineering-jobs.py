# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import textwrap

import plotly.graph_objects as go

import seaborn as sns
eng_jobs = pd.read_csv('../input/data-engineer-jobs/DataEngineer.csv')

eng_jobs.head(5)
eng_jobs.drop(columns=['Job Description','Company Name','Competitors'],axis=1,inplace=True)
eng_jobs.replace([-1.0,-1,'-1'],np.nan, inplace=True)

eng_jobs['Easy Apply'] = eng_jobs['Easy Apply'].fillna(False).astype(bool)
eng_jobs['Salary Estimate'] = eng_jobs['Salary Estimate'].str.replace('(','').str.replace(')','').str.replace('Glassdoor est.','').str.replace('Employer est.','')
eng_jobs['Mini Salary'],eng_jobs['Max Salary'] = eng_jobs['Salary Estimate'].str.split('-').str

eng_jobs['Mini Salary'] = eng_jobs['Mini Salary'].str.strip(' ').str.strip('$').str.strip('K').fillna(0).astype(int)

eng_jobs['Max Salary'] = eng_jobs['Max Salary'].str.strip(' ').str.strip('$').str.strip('K').fillna(0).astype(int)
eng_jobs.drop(columns=['Salary Estimate'],axis=1,inplace=True)
#Replaced the nan with the most occuring date

eng_jobs.Founded.replace(np.nan, 2000, inplace=True) 

eng_jobs['Founded'] = eng_jobs['Founded'].astype(int)
eng_jobs.Rating.replace(np.nan, 3.9, inplace=True)
eng_jobs['Revenue'].replace('Unknown / Non-Applicable',np.nan,inplace=True)

eng_jobs.head(5)
easy_sec = eng_jobs.loc[eng_jobs['Easy Apply'] == True]

easy_sec = easy_sec.groupby('Sector')['Easy Apply'].count().reset_index()

Easy_sec = easy_sec.sort_values('Easy Apply',ascending=False).head(8)
easy_sec.head(5)
fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = Easy_sec,x = 'Sector',y = 'Easy Apply', ax = ax)

ax.set_ylabel('Count Jobs')

ax.set_yticks(np.arange(0, 65, step = 5))

for index,Easy_sec in enumerate(Easy_sec['Easy Apply'].astype(int)):

       ax.text(x=index-0.1 , y =Easy_sec+1 , s=f"{Easy_sec}" , fontdict=dict(fontsize=10))

plt.show()
sala_city = eng_jobs.groupby('Location')[['Mini Salary','Max Salary']].mean().sort_values(['Mini Salary','Max Salary'],ascending=False)

sala_city.head(5)
fig = go.Figure()



fig.add_trace(go.Bar(x=sala_city.index,y=sala_city['Mini Salary'],name='Minimum salary'))

fig.add_trace(go.Bar(x=sala_city.index,y=sala_city['Max Salary'],name='Maximum Salary'))



fig.update_layout(title='Top 20 cities with their minimum and maximum salaries',barmode='stack')



fig.show()
Job_Rev = eng_jobs.groupby('Revenue')['Job Title'].count().reset_index()

Job_Rev.sort_values('Job Title',ascending=False, inplace=True)

Job_Rev.head(5)
max_width = 15

fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = Job_Rev,x = 'Revenue',y = 'Job Title', ax = ax)

ax.set_ylabel('Count Jobs')

ax.set_title('Job Title against Revenue')

ax.set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels())

for index,Job_Rev in enumerate(Job_Rev['Job Title'].astype(int)):

       ax.text(x=index-0.1 , y =Job_Rev+1 , s=f"{Job_Rev}" , fontdict=dict(fontsize=10))

plt.show()
rate_job = eng_jobs.groupby('Rating')['Job Title'].count().reset_index()

rate_job.sort_values('Job Title',ascending=False,inplace=True)

rate_job.head(5)
fig, ax = plt.subplots(figsize = (16,5))

sns.barplot(data = rate_job,x = 'Rating',y = 'Job Title', ax = ax)

ax.set_ylabel('Count Jobs')

ax.set_title('Rating against Job Title')

plt.show()
jobs = eng_jobs.loc[eng_jobs.Headquarters.isin(['Chicago, IL'])]

jobs.head(5)
own_sec = jobs.groupby('Type of ownership')['Sector'].count().reset_index()

own_sec.sort_values('Sector',ascending=False,inplace=True)

own_sec.head(5)
fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = own_sec,x = 'Type of ownership',y='Sector',ax = ax)

ax.set_ylabel('Count ownership')

ax.set_yticks(np.arange(0, 65, step = 5))

for index,own_sec in enumerate(own_sec['Sector'].astype(int)):

       ax.text(x=index-0.1 , y =own_sec+1 , s=f"{own_sec}" , fontdict=dict(fontsize=10))

plt.show()
money_min = jobs.groupby('Sector')[['Mini Salary','Max Salary']].mean().sort_values(['Mini Salary','Max Salary'],ascending=False).head(8)

money_min.reset_index(inplace=True)



money_max = jobs.groupby('Sector')[['Mini Salary','Max Salary']].mean().sort_values(['Mini Salary','Max Salary'],ascending=True).head(8)

money_max.reset_index(inplace=True)



print(money_max, '\n')

print(money_min)
max_width = 15

money = [money_min,money_max]

money_title = ['Top 8', 'Bottom 8']

fig, ax = plt.subplots(2,1, figsize = (22,14))

fig.subplots_adjust(hspace = 0.5)

for i in range(0,2):

    sns.barplot(ax = ax[i], data = money[i], x = 'Sector', y = 'Max Salary', color = 'orangered', label = 'Max Salary')

    sns.barplot(ax = ax[i], data = money[i], x = 'Sector', y = 'Mini Salary', color = 'darkslateblue', label = 'Mini Salary')

    ax[i].legend()

    ax[i].set_title(money_title[i]+' Average Salary in Each Sector', fontsize = 20)

    ax[i].set_ylabel('Salary', fontsize = 20)

    ax[i].set_xlabel('Sector', fontsize = 20)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 300, step = 50))

    ax[i].tick_params(labelsize = 18)

    

plt.show()