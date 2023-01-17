# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv')
df.head()
#dropping first column

df=df.drop(['Unnamed: 0'], axis=1)

df.head()
df.columns
df.shape
df.info()
df.isnull().sum()
df['course_Certificate_type'].unique()
df['course_difficulty'].unique()
#converting course_students_enrolled values to integers

df.course_students_enrolled.replace(r'[km]+$', '', regex=True).astype(float)

df['course_students_enrolled']=(df.course_students_enrolled.replace(r'[km]+$', '', regex=True).astype(float)*df.course_students_enrolled.str.extract(r'[\d\.]+([km]+)', expand=False).replace(['k','m'], [10**3, 10**6])).astype('int64')

df['course_students_enrolled']
fig_dims = (10,6)

fig,ax = plt.subplots(figsize=fig_dims)

sns.countplot(x='course_difficulty',data = df,ax=ax)
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data=df,x = 'course_Certificate_type',hue= 'course_difficulty',ax=ax)

plt.title('Count of course certificate types for each course difficulty')
fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data=df,x = 'course_rating',hue= 'course_difficulty',ax=ax)

plt.title('Count of course_ratings for course_difficulties')
fig = px.scatter(df, x="course_difficulty", y="course_organization", 

                 color="course_rating", 

                 hover_data=['course_title', 'course_rating'])

fig.show()
#Finding the total students enrolled for each course difficulty category

df1 = pd.DataFrame({'total_students_enrolled' : df.groupby('course_difficulty').sum()['course_students_enrolled']}).reset_index()

df1
my_colors = ['lightblue','lightsteelblue','silver','blue']

plt.pie(df1['total_students_enrolled'],labels=df1['course_difficulty'],autopct='%1.0f%%',shadow = True, colors=my_colors)

plt.title('Distribution of total students enrolled according to course difficulty')

plt.show()
fig = px.sunburst(df, path=['course_difficulty', 'course_organization'],values='course_rating')

fig.show()