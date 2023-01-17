# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
udemy_data = pd.read_csv("/kaggle/input/udemy-courses/udemy_courses.csv")

udemy_data.head()
# Change into boolean form

udemy_data.is_paid.replace(['FALSE', 'https://www.udemy.com/learnguitartoworship/'], 'False', inplace = True)

udemy_data.is_paid.replace('TRUE', 'True', inplace = True)

udemy_data.level.replace('52', 'All Levels', inplace = True)

udemy_data = udemy_data.drop_duplicates().reset_index(drop=True)
udemy_data.info
# Drop the columns that are irrelevant



udemy_data.drop(['course_id','url'], axis=1, inplace=True)
subscribers = udemy_data.groupby('is_paid')['num_subscribers'].agg('sum').to_frame()

fig = px.pie(subscribers, values='num_subscribers', names= ['Paid', 'UnPaid'], title='Subscribers Correlation Chart')

fig.show()

subjects = udemy_data.groupby('subject')['num_subscribers'].agg('sum').to_frame()

names = udemy_data['subject'].unique()

fig = px.pie(subjects, values='num_subscribers', names= names , title='Subscribers and Subject Correlation Chart')

fig.update_traces(rotation=90)

fig.show()
# Converting the number of lectures into a range

bins = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, np.inf]

bin_names =  ['<25', '25h-50', '50h-75', '75h-100', '100h-150', '150h-200', '200h-250', '250h-300', '300h-350', '350h-400', '400h-450', '450h-500', '500+']

udemy_data['lectures_range'] = pd.cut(udemy_data['num_lectures'], bins, labels=bin_names)
# Using the lectures_range column to find the correlation to number of subscribers

lectures = udemy_data.groupby('lectures_range')['num_subscribers'].agg('sum').to_frame()

fig = px.pie(lectures, values='num_subscribers', names= bin_names , title='Subscribers and Number of Lectures Correlation Chart')

fig.show()

level = udemy_data.groupby('level')['num_subscribers'].agg('sum').to_frame()

names = udemy_data['level'].unique()

fig = px.pie(level, values='num_subscribers', names= sorted(names) , title='Subscribers and Course Level Correlation Chart')

fig.show()

# Converting the number of lectures into a range

bins = [0, 5, 10, 15, 20, np.inf]

bin_names =  ['0-5h', '5-10h', '10-15h', '15-20h', '20h+']

udemy_data['content_range'] = pd.cut(udemy_data['content_duration'], bins, labels=bin_names)
# Using the content_range column to find the correlation to number of subscribers

lectures = udemy_data.groupby('content_range')['num_subscribers'].agg('sum').to_frame()

fig = px.pie(lectures, values='num_subscribers', names= bin_names , title='Subscribers and Content duration Correlation Chart')

fig.show()
subjects = udemy_data.groupby('subject')['num_reviews'].sum().to_frame()

names = udemy_data['subject'].unique()



fig = px.pie(subjects, values='num_reviews', names= names , title='Reviews for each subject')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
paid_courses = udemy_data.loc[udemy_data['is_paid'] == True, ['price', 'subject']]

most_paid = udemy_data.groupby('subject')['price'].sum().reset_index()

names = paid_courses['subject'].unique()



fig = px.pie(most_paid, values='price', names= names , title='Most high paid Courses')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
paid_courses = udemy_data.loc[udemy_data['is_paid'] == True, ['is_paid', 'subject']]

most_paid = udemy_data.groupby('subject')['is_paid'].count().reset_index()

names = paid_courses['subject'].unique()



fig = px.pie(most_paid, values='is_paid', names= names , title='Most paid Courses')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
free_courses = udemy_data.loc[udemy_data['is_paid'] == False, ['is_paid', 'subject']]

most_free = free_courses.groupby('subject')['is_paid'].count().to_frame()

names = free_courses['subject'].unique()



fig = px.pie(most_free, values='is_paid', names= names , title='Most free Courses')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
courses = udemy_data.loc[:, ['course_title','num_reviews']]

most_reviews = courses.groupby('course_title')['num_reviews'].max().reset_index()

most_reviews_sort = most_reviews.sort_values(by = 'num_reviews', ascending = False)

most_reviews_sort = most_reviews_sort.head(10)



fig = px.pie(most_reviews_sort, values='num_reviews', names= 'course_title' , title='Most Reviews for Course titles', template = 'seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
courses = udemy_data.loc[:, ['course_title','num_reviews']]

least_reviews = courses.groupby('course_title')['num_reviews'].max().reset_index()

least_reviews = least_reviews[least_reviews['num_reviews'] > 10]

least_reviews_sort = least_reviews.sort_values(by = 'num_reviews')

least_reviews_sort = least_reviews_sort.head(10)



fig = px.pie(least_reviews_sort, values='num_reviews', names= 'course_title' , title='Least Reviews for Course titles', template = 'seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
courses = udemy_data.loc[:, ['course_title','num_subscribers']]

most_subs = courses.groupby('course_title')['num_subscribers'].max().reset_index()

most_subs_sort = most_subs.sort_values(by = 'num_subscribers', ascending = False)

most_subs_sort = most_subs_sort.head(10)



fig = px.pie(most_subs_sort, values='num_subscribers', names= 'course_title' , title='Most subscribers for Course titles', template = 'seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
courses = udemy_data.loc[:, ['course_title','num_subscribers']]

least_subs = courses.groupby('course_title')['num_subscribers'].max().reset_index()

least_subs = least_subs[least_subs['num_subscribers'] > 10]

least_subs_sort = least_subs.sort_values(by = 'num_subscribers')

least_subs_sort = least_subs_sort.head(10)



fig = px.pie(least_subs_sort, values='num_subscribers', names= 'course_title' , title='Least subscribers for Course titles', template = 'seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()