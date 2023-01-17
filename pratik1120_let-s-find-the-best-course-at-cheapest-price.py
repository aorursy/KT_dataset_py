import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go



data = pd.read_csv('../input/udemy-courses/udemy_courses.csv')

data.drop(['course_id','url'], axis=1, inplace=True)
data['price'] = data['price'].str.replace('Free','0')

data['price'] = data['price'].str.replace('TRUE','0')        

data['price'] = data['price'].astype('int')



data['is_paid'] = data['is_paid'].replace('TRUE', 'True')

data['is_paid'] = data['is_paid'].replace('FALSE', 'False')

data['is_paid'] = data['is_paid'].replace('https://www.udemy.com/learnguitartoworship/', 'True')
subject = data.loc[:,['price', 'subject']]

subject['total_price'] = subject.groupby('subject')['price'].transform('sum')

subject.drop(['price'],axis=1, inplace=True)

subject = subject.drop_duplicates().reset_index(drop=True)

subject = subject.sort_values('total_price')



fig = px.pie(subject, names='subject', values='total_price', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
subject = data.loc[data['is_paid']=='False', ['is_paid','subject']]

subject['count'] = subject.groupby('subject')['is_paid'].transform('count')

subject.drop(['is_paid'],axis=1, inplace=True)

subject = subject.drop_duplicates().reset_index(drop=True)

subject = subject.sort_values('count')



fig = px.pie(subject, names='subject', values='count', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
sns.kdeplot(data=data['price'], label='price', shade=True)
subject = data.loc[data['is_paid']=='True', ['is_paid','subject']]

subject['count'] = subject.groupby('subject')['is_paid'].transform('count')

subject.drop(['is_paid'],axis=1, inplace=True)

subject = subject.drop_duplicates().reset_index(drop=True)

subject = subject.sort_values('count')



fig = px.pie(subject, names='subject', values='count', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
sns.kdeplot(data=data['num_subscribers'], label='num_subscribers', shade=True)
course = data.loc[:, ['course_title','num_subscribers']]

course = course.groupby('course_title')['num_subscribers'].max().reset_index()

course = course.sort_values('num_subscribers', ascending=False)

course = course.head(10)



fig = px.pie(course, names='course_title', values='num_subscribers', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
course = data.loc[(data['num_subscribers']>100), ['course_title','num_subscribers']]

course = course.groupby('course_title')['num_subscribers'].min().reset_index()

course = course.sort_values('num_subscribers', ascending=False)

course = course.tail(10)



fig = px.pie(course, names='course_title', values='num_subscribers', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
course = data.loc[:, ['course_title','num_reviews']]

course = course.groupby('course_title')['num_reviews'].max().reset_index()

course = course.sort_values('num_reviews', ascending=False)

course = course.head(7)



fig = go.Figure(data=[go.Pie(labels=course['course_title'], values=course['num_reviews'])])

fig.update_traces(rotation=90, pull=[0.2,0.05,0.05,0.05,0.05,0.05,0.05], textinfo="percent+label")

fig.show()
course = data.loc[data['num_reviews']>10, ['course_title','num_reviews']]

course = course.groupby('course_title')['num_reviews'].min().reset_index()

course = course.sort_values('num_reviews', ascending=False)

course = course.tail(10)



fig = px.pie(course, names='course_title', values='num_reviews', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
course = data.loc[:, ['level','num_subscribers']]

course['count'] = course.groupby('level')['num_subscribers'].transform('sum')

course = course.sort_values('count', ascending=False)

course.drop('num_subscribers', axis=1, inplace=True)

course = course.drop_duplicates().reset_index(drop=True)

course = course.head()



fig = plt.figure(figsize=(10,7))

sns.barplot(data=course, x='level', y='count')
course = data.loc[:, ['course_title','num_lectures']]

course = course.groupby('course_title')['num_lectures'].max().reset_index()

course = course.sort_values('num_lectures', ascending=False)

course = course.head()



fig = plt.figure(figsize=(10,7))

plt.pie(course['num_lectures'], labels=course['course_title'], autopct='%1.1f%%', shadow=True)

centre_circle = plt.Circle((0,0),0.45,color='black', fc='white',linewidth=1.25)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.legend(loc='center right')

plt.axis('equal')

plt.show()
course = data.loc[data['is_paid']=='True', ['course_title','num_subscribers']]

course = course.groupby('course_title')['num_subscribers'].max().reset_index()

course = course.sort_values('num_subscribers', ascending=False)

course = course.head(10)



fig = px.pie(course, names='course_title', values='num_subscribers', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
top_5 = course['course_title'].head().tolist()
course = data.loc[data['course_title'].isin(top_5), ['course_title','num_reviews']]

course = course.groupby('course_title')['num_reviews'].max().reset_index()

course = course.sort_values('num_reviews', ascending=False)

course = course.head()



fig = go.Figure(data=[go.Pie(labels=course['course_title'], values=course['num_reviews'])])

fig.update_traces(rotation=90, pull=[0.2,0.05,0.05,0.05,0.05], textinfo="percent+label")

fig.show()
course = data.loc[data['course_title'].isin(top_5), ['course_title','num_lectures']]

course = course.groupby('course_title')['num_lectures'].max().reset_index()

course = course.sort_values('num_lectures', ascending=False)

course = course.head()



fig = go.Figure(data=[go.Pie(labels=course['course_title'], values=course['num_lectures'])])

fig.update_traces(rotation=90, pull=[0.2,0.05,0.05,0.05,0.05], textinfo="percent+label")

fig.show()
def content(strng):

    lst = strng.split()

    return float(lst[0])

top_5_courses = data[data['course_title'].isin(top_5)]

top_5_courses['content_duration'] = top_5_courses['content_duration'].apply(content)
course = top_5_courses.loc[:, ['course_title','content_duration']]

course = course.groupby('course_title')['content_duration'].max().reset_index()

course = course.sort_values('content_duration', ascending=False)

course = course.head()



plt.figure(figsize=(10,7))

sns.barplot(data=course, x='content_duration', y='course_title')

plt.show()
course = data.loc[data['is_paid']=='False', ['course_title','num_subscribers']]

course = course.groupby('course_title')['num_subscribers'].max().reset_index()

course = course.sort_values('num_subscribers', ascending=False)

course = course.head()



fig = px.pie(course, names='course_title', values='num_subscribers', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
top_5 = course['course_title'].head().tolist()



def content_free(strng):

    lst = strng.split()

    return float(lst[0])



top_5_courses = data[data['course_title'].isin(top_5)]

top_5_courses['content_duration'] = top_5_courses['content_duration'].apply(content_free)
course = top_5_courses.loc[:, ['course_title','num_reviews']]

course = course.groupby('course_title')['num_reviews'].max().reset_index()

course = course.sort_values('num_reviews', ascending=False)

course = course.head()



fig = go.Figure(data=[go.Pie(labels=course['course_title'], values=course['num_reviews'])])

fig.update_traces(rotation=90, pull=[0.2,0.05,0.05,0.05,0.05], textinfo="percent+label")

fig.show()
course = top_5_courses.loc[:, ['course_title','num_lectures']]

course = course.groupby('course_title')['num_lectures'].max().reset_index()

course = course.sort_values('num_lectures', ascending=False)

course = course.head()



plt.figure(figsize=(10,7))

sns.barplot(data=course, x='num_lectures', y='course_title')

plt.show()