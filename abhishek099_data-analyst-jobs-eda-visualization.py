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
data = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')
data.head(10)
data = data.drop('Unnamed: 0', axis = 1)
data.isnull().sum()
data.fillna(0)
data.info()
data.describe()
data = data.replace(-1, np.nan)
data = data.replace(-1.0, np.nan)
data = data.replace('-1', np.nan)
data.isnull().sum()

data['Company Name'],_ = data['Company Name'].str.split('\n',1).str
data['Job Title'], data['Department'] = data['Job Title'].str.split(',', 1).str
data['Salary Estimate'],_ = data['Salary Estimate'].str.split('(', 1).str
data.head(3)
data['Min Salary'], data['Max Salary'] = data['Salary Estimate'].str.split('-', 1).str
data['Min Salary'] = data['Min Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype(int)
data['Max Salary'] = data['Max Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype(int)
data = data.drop('Salary Estimate', axis = 1)
data.head(3)
data['Easy Apply'] = data['Easy Apply'].fillna(False).astype(bool)
df_easy_apply = data[data['Easy Apply'] == True]
df = df_easy_apply.groupby('Company Name')['Easy Apply'].count().reset_index()
company_opening_df = df.sort_values('Easy Apply', ascending = False)

plt.figure(figsize = (20, 10))
sns.barplot(x = company_opening_df['Company Name'], y = company_opening_df['Easy Apply'])
plt.xticks(rotation = 90)
data_analyst = data[data['Job Title'] == 'Data Analyst']

f, ax = plt.subplots(1, 2, sharex = True, figsize = (15,8))
sns.distplot(data_analyst['Min Salary'], color = 'b', ax = ax[0])

sns.distplot(data_analyst['Max Salary'], color = 'r', ax = ax[1])
plt.setp(ax, yticks = [])
df = data.groupby('Location')[['Min Salary', 'Max Salary']].mean().sort_values(['Min Salary', 'Max Salary'], ascending = False)
df
import plotly.graph_objects as go
fig = go.Figure()

fig.add_trace(go.Bar(x = df.index[:20], y = df['Min Salary'][:20], name = 'Minimum Salary'))
fig.add_trace(go.Bar(x = df.index[:20], y = df['Max Salary'][:20], name = 'Maximum Salary'))
fig.show()
df = data.groupby('Job Title')[['Min Salary', 'Max Salary']].mean().sort_values(['Min Salary', 'Max Salary'], ascending = False)
df.head(20)
fig = go.Figure()

fig.add_trace(go.Bar(x = df.index[:20], y = df['Min Salary'][:20], name = 'Minimum Salary'))
fig.add_trace(go.Bar(x = df.index[:20], y = df['Max Salary'][:20], name = 'Maximum Salary'))
fig.show()
plt.figure(figsize = (10,10))
sns.countplot(data['Size'])
plt.xticks(rotation = 65)
plt.ylabel('No. of Comapnies')
data['Min Revenue'], data['Max Revenue'] = data['Revenue'].str.split('to', 1).str
data['Max Revenue'] = data['Max Revenue'].str.strip(' ').str.rstrip('million (USD)').str.rstrip('billion (USD)').str.lstrip('$').fillna(0).astype(int)

df = data.groupby('Sector')[['Max Revenue']].mean().sort_values(['Max Revenue'], ascending = False)
df.head(20)
plt.figure(figsize = (15, 8))
sns.barplot(y = data['Max Revenue'], x = data['Sector'])
plt.xticks(rotation = 90)
from wordcloud import WordCloud
job_title = data['Job Title'][~pd.isnull(data['Job Title'])]
wordCloud = WordCloud(width=450,height= 300).generate(' '.join(job_title))
plt.figure(figsize = (19,9))
plt.axis('off')
plt.title(data['Job Title'].name,fontsize = 20)
plt.imshow(wordCloud)
plt.show()
plt.figure(figsize = (12,12))
sns.countplot(sorted(data['Rating'], reverse = False))
plt.xticks(rotation = 315)
plt.ylabel('No. of Companies')
df = data.groupby('Industry')[['Min Salary', 'Max Salary']].mean().sort_values(['Min Salary', 'Max Salary'], ascending = False).head(20)
df = df.reset_index()
df.head()
fig = go.Figure()

fig.add_trace(go.Bar(x = df['Industry'], y = df['Min Salary'], name = 'Average Min Salary'))
fig.add_trace(go.Bar(x = df['Industry'], y = df['Max Salary'], name = 'Average Max Salary'))

df = data[data['Easy Apply'] == True]
job_openings = df.groupby('Job Title')[['Easy Apply']].count()
job_openings = job_openings.sort_values('Easy Apply', ascending = False)
job_openings = job_openings.reset_index()
job_openings
plt.figure(figsize = (10,5))
sns.barplot(x = job_openings['Job Title'][:10], y = job_openings['Easy Apply'][:10])
plt.xticks(rotation = 65)
plt.ylabel('Job Openings')