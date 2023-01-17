import pandas as pd

import numpy as np

import re

from sklearn.preprocessing import LabelEncoder

from wordcloud import WordCloud

from collections import Counter

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data.head()
data.info()
data = data.replace(-1, np.nan)

data = data.replace('-1', np.nan)

data = data.replace(-1.0, np.nan)
data.info()
data['Salary Estimate'].value_counts()
data['Salary Estimate'] = data['Salary Estimate'].apply(lambda x: str(x))

data['Salary Estimate'] = data['Salary Estimate'].apply(lambda x: re.findall(r'\d+', x))

data['sal_est_min'] = data['Salary Estimate'].str[0].astype(np.float32)

data['sal_est_max'] = data['Salary Estimate'].str[1].astype(np.float32)

data.drop('Salary Estimate', axis=1, inplace=True)
data['Revenue'].value_counts()
data['rev_scale'] = data['Revenue'].apply(lambda x: 1e6 if 'million' in str(x) else (1e9 if 'billion' in str(x) else (np.nan)))

data['rev_scale'] = data['rev_scale'].astype(np.float32)

data['Revenue'] = data['Revenue'].apply(lambda x: str(x))

data['Revenue'] = data['Revenue'].apply(lambda x: re.findall(r'\d+', x))

data['rev_min'] = data['Revenue'].str[0].astype(np.float32) * data['rev_scale']

data['rev_max'] = data['Revenue'].str[1].astype(np.float32) * data['rev_scale']

data.drop(['Revenue', 'rev_scale'], axis=1, inplace=True)
data['Size'].value_counts()
data['Size'] = data['Size'].apply(lambda x: str(x))

data['Size'] = data['Size'].apply(lambda x: re.findall(r'\d+', x))

data['min_emp_size'] = data['Size'].str[0].astype(np.float32)

data['max_emp_size'] = data['Size'].str[1].astype(np.float32)

data.drop('Size', axis=1, inplace=True)
data['Location'].value_counts()
data['city'], data['state'] = data['Location'].str.split(',', 1).str

data.drop('Location', axis=1, inplace=True)
data['Headquarters'].value_counts()
data['hq_city'], data['hq_state'] = data['Headquarters'].str.split(',',1).str

data.drop('Headquarters', axis=1, inplace=True)
data['company'] = data['Company Name'].apply(lambda x: str(x).split('\n')[0])

data.drop('Company Name', axis=1, inplace=True)
data['Job Title'].head()
data['Job Title'], data['Department'] = data['Job Title'].str.split(',', 1).str

data[['Job Title', 'Department']].head()
data.drop('Unnamed: 0', axis=1, inplace=True)
data.head()
job_titles = list(data['Job Title'].values)

jt = ' '.join(job_titles)

wc = WordCloud(width = 800, height = 640, background_color='black').generate(jt)

plt.figure(figsize = (14, 12))

plt.imshow(wc)

plt.axis('off')

plt.show()
titles , values = zip(*Counter(job_titles).most_common(20))

plt.figure(figsize=(12,7))

plt.bar(titles, values)

plt.title('Top Job Titles')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(data['Founded'], bins=100, kde=False, color='orange', hist_kws={"rwidth":0.75,'edgecolor':'black', 'alpha':1.0})

plt.xlim([1930, 2020])

plt.title('Founding Year')

plt.xlabel('Year Founded')

plt.ylabel('No .of companies')

plt.show()
ownership= list(data['Type of ownership'].dropna().values)

dict1 = Counter(ownership)

dict1 = sorted(dict1.items(), key=lambda x: x[1], reverse=True)

ownership_type, counts = zip(*dict1)

plt.figure(figsize=(12,7))

plt.bar(ownership_type, counts, color='crimson')

plt.title('Ownership Type')

plt.xticks(rotation=90)

plt.xlabel('Sector')

plt.ylabel('No. of companies')

plt.show()
industry = data['Industry'].dropna().values

ind, vals = zip(*Counter(industry).most_common(15))

plt.figure(figsize=(12,8))

plt.bar(ind, vals, color='brown')

plt.xticks(rotation=90)

plt.xlabel('Industry')

plt.ylabel('No. of companies')

plt.show()
sector = data['Sector'].dropna()

fig = px.pie(sector, names='Sector', title='Sectors')

fig.show()
# Total no. of cities in data

len(data['city'].unique())
top_25 = data.groupby('city')[['sal_est_min', 'sal_est_max']].mean().sort_values(by = ['sal_est_max','sal_est_min'], ascending = False).head(25)



fig = go.Figure()



fig.add_trace(go.Bar(x=top_25.index, y=top_25.sal_est_min, name='Min Salary'))

fig.add_trace(go.Bar(x=top_25.index, y=top_25.sal_est_max, name='Max Salary'))



fig.update_layout(title='Min and Max salaries in top 25 cities for Data Analyst roles in $\'000', barmode='stack')

fig.show()
roles_sal = data.groupby('Job Title')[['sal_est_min', 'sal_est_max']].mean().sort_values(['sal_est_max', 'sal_est_min'], ascending=False).head(20)



fig = go.Figure()



fig.add_trace(go.Bar(x=roles_sal.index, y=roles_sal.sal_est_min, name='Min Salary'))

fig.add_trace(go.Bar(x=roles_sal.index, y=roles_sal.sal_est_max, name='Max Salary'))



fig.update_layout(title='Min & Max salaries of 20 Job titles in $\'000', barmode='stack')

fig.show()
data['Easy Apply'][data['Easy Apply'].notna()].shape[0]/data.shape[0] * 100
openings = data['Job Title'][data['Easy Apply'] == 'True']

openings = Counter(openings).most_common(15)

openings, counts = zip(*openings)

plt.figure(figsize=(12,6))

plt.bar(openings, counts)

plt.title('Job vacancies')

plt.xticks(rotation=90)

plt.xlabel('Job Title')

plt.ylabel('No. of vacancies')

plt.show()