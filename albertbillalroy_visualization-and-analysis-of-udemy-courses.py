import numpy as np

import pandas as pd

import plotly.express as px
df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')
df.head()
missing_values_count = df.isnull().sum()



# nan by columns

missing_values_count
# removing duplicates

df.drop_duplicates()
# replace value of is_paid column

mask = df.applymap(type) != bool

d = {True: 'Paid', False: 'Free'}

df = df.where(mask, df.replace(d))
# rename subject column to be categories

df = df.rename(columns={"subject": "categories"})
# print the total number of unique categories

num_categories = df['categories'].nunique()

print('Number of categories = ', num_categories)
# count the number of apps in each 'Category' and sort them for easier plotting

df.groupby('categories')['course_id'].count()



fig = px.bar(df.groupby('categories')['course_id'].count(), x=(df.groupby('categories')['course_id'].count()).values, y=(df.groupby('categories')['course_id'].count()).index, color=(df.groupby('categories')['course_id'].count()).index, orientation='h')

fig.update_layout(

    title="Total courses in each category",

    xaxis_title="Total courses",

    yaxis_title="Category"

)

fig.show()
(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False)
fig = px.bar((df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False), x=(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False).index, y=(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False).values, text=(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False).values, color=(df.groupby('is_paid')['course_id'].count()).sort_values(ascending=False).index)

fig.update_layout(

    title="Number of paid courses vs free courses",

    xaxis_title="Type",

    yaxis_title="Total Courses"

)

fig.show()
(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False)
fig = px.bar((df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False), x=(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False).index, y=(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False).values, text=(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False).values, color=(df.groupby('is_paid')['num_subscribers'].sum()).sort_values(ascending=False).index)

fig.update_layout(

    title="Number of paid subscribers vs free subscribers",

    xaxis_title="Type",

    yaxis_title="Total subscribers"

)

fig.show()
fig = px.bar(df.groupby('price')['course_id'].count(), y=(df.groupby('price')['course_id'].count()).values, x=(df.groupby('price')['course_id'].count()).index, text=(df.groupby('price')['course_id'].count()).values)

fig.update_layout(title="Price distribution",

                 xaxis_title="Price",

                 yaxis_title="Frequency"

                 )

fig.show()
fig = px.scatter(df, x="price", y="num_subscribers", color="categories")

fig.update_layout(title="Course price and number of subscribers",

                 xaxis_title="Price",

                 yaxis_title="Number of subscribers"

                 )

fig.show()
fig = px.scatter(df, x="num_reviews", y="num_subscribers", color="categories")

fig.update_layout(title="Course number of reviews and number of subscribers",

                 xaxis_title="Number of reviews",

                 yaxis_title="Number of subscribers"

                 )

fig.show()
fig = px.scatter(df, x="content_duration", y="num_subscribers", color="categories")

fig.update_layout(title="Course content duration and number of subscribers",

                 xaxis_title="Content duration",

                 yaxis_title="Number of subscribers"

                 )

fig.show()
fig = px.scatter(df, x="published_timestamp", y="num_subscribers", color="categories")

fig.update_layout(title="Year of course and number of subscribers",

                 xaxis_title="Year",

                 yaxis_title="Number of subscribers"

                 )

fig.show()