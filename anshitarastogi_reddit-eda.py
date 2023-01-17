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
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go
data=pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')

data.head()
data.isnull().sum()
data.shape
print('Unique values in author_flair_text',data['author_flair_text'].unique(),'\n')

print('Unique values in removed_by',data['removed_by'].unique(),'\n')

print('Unique values in total_awards_received',data['total_awards_received'].unique(),'\n')

print('Unique values in awarders',data['awarders'].unique(),'\n')
data=data.drop(['author_flair_text','awarders'],axis=1)

data.head()
nan_replacements = { "removed_by": 'unknown', "title": 'unknown','total_awards_received':0.0}

data = data.fillna(nan_replacements)

data
data.info()
data['formatted_created_utc']=pd.to_datetime(data['created_utc'],unit='s')

data


author=data['author'].value_counts().head()

author_count=pd.DataFrame(author.items(), columns=['Name','Count'])

author_count
fig=px.bar(author_count,x='Name',y='Count')

fig.show()
number_of_awards_received=data['total_awards_received'].value_counts()

number_of_awards_received_df=pd.DataFrame(number_of_awards_received.items(),columns=['Award Count','Author Count'])

number_of_awards_received_df=number_of_awards_received_df.iloc[1:,:]

figure=px.bar(number_of_awards_received_df,x='Award Count',y='Author Count')

figure.show()

print('Minimum score = ',data['score'].min(),' and maximum score = ',data['score'].max())

print('Unique values in score column is : ',data['score'].unique())
data['score'].value_counts()
sns.kdeplot(data.score)

plt.xlabel("Score")

plt.ylabel("Freq")
most_pop_num_comments=data.sort_values('num_comments',ascending=False)[['title','score','author','full_link','num_comments']].head()

fig2=px.bar(most_pop_num_comments,x='title',y='num_comments',title='Best Reddit Post based on maximum number of comments',hover_data=['author', 'full_link'])

fig2.show()
most_pop_score=data.sort_values('score',ascending=False)[['title','score','author','full_link','num_comments']].head()

fig3=px.bar(most_pop_score,x='title',y='score',title='Best Reddit Post based on maximum score',hover_data=['author', 'full_link'])

fig3.show()
print(data['over_18'].value_counts())
from datetime import datetime

data['Day']=data['formatted_created_utc'].dt.day

data['Month']=data['formatted_created_utc'].dt.month

data['Year']=data['formatted_created_utc'].dt.year

data['date_utc']=data['formatted_created_utc'].dt.date

data.head()
post_posted_everday=data.groupby('date_utc')['title'].count().reset_index()

post_posted_everday
max_post_posting_date=post_posted_everday.sort_values(by='title',ascending=False).rename(columns={'title':'Post Count'}).head(10)

fig4=px.bar(max_post_posting_date,x='date_utc',y='Post Count',title='Post Count based on date')

fig4.show()
post_in_years=data.groupby('Year')['title'].count().reset_index().rename(columns={'title':'Post Count'})

post_in_years=post_in_years.sort_values(by='Post Count',ascending=False)

fig5=px.bar(post_in_years,x='Year',y='Post Count',title='Post Count based on years')

fig5.show()
post_in_months=data.groupby('Month')['title'].count().reset_index().rename(columns={'title':'Post Count'})

post_in_months=post_in_months.sort_values(by='Post Count',ascending=False)

fig6=px.bar(post_in_months,x='Month',y='Post Count',title='Post Count based on months')

fig6.show()
post_in_days=data.groupby('Day')['title'].count().reset_index().rename(columns={'title':'Post Count'})

post_in_days=post_in_days.sort_values(by='Post Count',ascending=False)

fig7=px.bar(post_in_days,x='Day',y='Post Count',title='Post Count based on days')

fig7.show()
post_in_days_months=data.groupby(['Month','Day'])['id'].count().reset_index().rename(columns={'id':'Post Count'})

post_in_days_months=post_in_days_months.sort_values(by='Post Count',ascending=False)

fig8=px.bar(post_in_days_months,x='Day',y='Post Count',title='Post Count based on days and months', color='Month')

fig8.show()
plt.figure(figsize=(15,10))

ax = sns.boxplot(data=post_in_days_months, x='Month', y='Post Count', 

                 showfliers=False, color='yellow', linewidth=2

                )



sns.despine(offset=10, trim=True)

ax.set(xlabel='Month', ylabel='Posts')

plt.title('Distribution of monthly post from 2012 till 2020', size=15)

plt.show()

post_in_months_year=data.groupby(['Month','Year'])['title'].count().reset_index().rename(columns={'title':'Post Count'})

post_in_months_year=post_in_months_year.sort_values(by='Post Count',ascending=False)

fig9=px.bar(post_in_months_year,x='Year',y='Post Count',title='Post Count based on days', color='Month')

fig9.show()
data.to_csv('submission1.csv', index=False)