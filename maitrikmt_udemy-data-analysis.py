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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go
#import the Dataset

udemy=pd.read_csv("../input/udemy-courses/udemy_courses.csv")
udemy.columns
udemy.info()
udemy.describe()
udemy.shape
# there is no null values

udemy.isnull().sum()
#Remove the irrelevant columns

udemy.drop("url",axis=1,inplace=True)
udemy
plt.figure(figsize=(10,9))

labels = ['Paid', 'Free']

size=udemy['is_paid'].value_counts()

print(size)



plt.rcParams['figure.figsize'] = (6, 6)

plt.pie(size,labels=labels,explode=[0,0.05],shadow=False,autopct='%.2f%%')

plt.title('Analysis of Course Type', fontsize = 16)

plt.axis('off')

plt.legend()

plt.show()
plt.figure(figsize=(10,9))

labels=['Web Development','Business Finance','Musical Instruments','Graphic Design']

size=udemy.subject.value_counts()

print(size)

plt.title("Analysis of subject",fontsize=16)

plt.pie(size,labels=labels,autopct='%.2f%%')

plt.legend()

plt.show()
labels=['All Levels','Beginner Level','Intermediate Level','Expert Level']

plt.figure(figsize =(10,9))

size=udemy.level.value_counts()

print(size)

plt.pie(size,labels=labels,explode=[0.01,0.02,0.03,0.04],autopct='%.2f%%')

plt.show()
udemy['year']=udemy['published_timestamp'].str[0:4]

udemy
plt.figure(figsize=(8,5))



bins=[2011,2012,2013,2014,2015,2016,2017]

year_count=udemy.year.value_counts().sort_index()

print(year_count)

plt.plot(year_count.index,year_count.values,'go-')

plt.title("course uploaded in each year",fontsize=19)

plt.xlabel("Year",fontsize=14)

plt.ylabel("course",fontsize=14)

plt.show()
course_free_lecture=udemy[udemy['is_paid']==False]['num_lectures'].sum()

course_paid_lecture=udemy[udemy['is_paid']==True]['num_lectures'].sum()



plt.figure(figsize=(10,5))

fig=go.Figure()

fig.add_trace(go.Bar(

        x=["Free","Paid"],

        y=[course_free_lecture,course_paid_lecture],

        width=[0.3,0.3]

))

fig.update_layout(title="Course Type (Free /paid)")

fig.show()
plt.figure(figsize = (10,5))

sns.countplot(x='subject',data=udemy,hue='is_paid')
top_10_course=udemy.sort_values(by='num_subscribers',ascending=False).head(10)
top_10_course
fig = go.Figure(data=[

    go.Bar(name='Subscribers', y=top_10_course['num_subscribers'], x=top_10_course['course_title']),

])

fig.update_layout(title="Top 10 courses")

fig.show()
fig=go.Figure(data=[

    go.Bar(name='Subscribers', x=top_10_course['course_title'], y=top_10_course['num_subscribers'],text=top_10_course['num_subscribers'],textposition='auto'),

    go.Bar(name='Reviews', x=top_10_course['course_title'], y=top_10_course['num_reviews'],text=top_10_course['num_reviews'],textposition='auto')

])



fig.update_layout(barmode='group',xaxis_tickangle=-30,title="Top 10 courses subscribers and reviews")

fig.show()
plt.rcParams['figure.figsize'] = (7, 5)



plt.subplot(1, 1, 1)

sns.set(style = 'whitegrid')

sns.distplot(udemy['price'])

plt.title('Distribution of price', fontsize = 16)

plt.xlabel('Range of price')

plt.ylabel('Count')
bins =[0,1,2,3,4,5,6,7,8]



#plotting distribution

plt.figure(figsize=(10,5))



plt.title('distribution of course duration (in hours)',fontsize=19)



plt.hist(udemy['content_duration'], bins=bins )

plt.xticks(bins)



plt.xlabel("duration of content (in hours)",fontsize=14)

plt.ylabel("no. of courses",fontsize=14)



plt.show()