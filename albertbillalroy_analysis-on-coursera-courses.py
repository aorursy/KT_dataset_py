import numpy as np

import pandas as pd

import plotly.express as px
df = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')
df.head()
missing_values_count = df.isna().sum()



# nan by columns

missing_values_count
# removing duplicates

df.drop_duplicates()
df = df.rename(columns={"Unnamed: 0": "course_id"})
df['course_students_enrolled']
def func(a):

    if 'k' in a:

        return float(str(a).replace('k', '')) * (10 ** 3)

    if 'm' in a:

        return float(str(a).replace('m', '')) * (10 ** 6)

    else:

        return float(a)
df['course_students_enrolled'] = df['course_students_enrolled'].apply(func)

df.head()
fig = px.bar(df['course_Certificate_type'].value_counts(), x=(df['course_Certificate_type'].value_counts()).index, y=(df['course_Certificate_type'].value_counts()).values, color=(df['course_Certificate_type'].value_counts()).index)

fig.update_layout(title_text="Number of movies each certificate",

                 xaxis_title="Certificate",

                 yaxis_title="Count")

fig.show()
course_org_df = pd.DataFrame(df.groupby('course_organization')['course_id'].count())



course_org_df = course_org_df.sort_values(by='course_id', ascending=False)[:20]



course_org_df = course_org_df.reset_index()



course_org_df = course_org_df.rename(columns={"course_id": "total_courses"})
fig = px.bar(course_org_df, x='total_courses', y='course_organization', color='course_organization', orientation='h',

             height=550)

fig.update_layout(title_text='20 Course organization with the most course title',

                 xaxis_title="Number of course",

                 yaxis_title="Course organization")

fig.show()
temp_df = pd.DataFrame(df.groupby('course_organization')['course_rating'].mean())



temp_df = temp_df.reset_index()
course_org_df = course_org_df.merge(temp_df, on='course_organization', how='inner')
fig = px.bar(course_org_df.sort_values(by='course_rating', ascending=False), x='course_rating', y='course_organization', color='course_organization', orientation='h',

             height=550)

fig.update_layout(title_text='20 highest rated course organization',

                 xaxis_title="Ratings",

                 yaxis_title="Course organization")

fig.show()
temp2_df = pd.DataFrame(df.groupby('course_organization')['course_students_enrolled'].sum())



temp2_df = temp2_df.reset_index()
course_org_df = course_org_df.merge(temp2_df, on='course_organization', how='inner')
fig = px.bar(course_org_df.sort_values(by='course_students_enrolled', ascending=False), x='course_students_enrolled', y='course_organization', color='course_organization', orientation='h',

             height=550)

fig.update_layout(title_text='20 most popular course organization',

                 xaxis_title="Number of Students",

                 yaxis_title="Course organization")

fig.show()
fig = px.histogram(df, x="course_rating")

fig.update_layout(title_text="Rating distribution",

                 xaxis_title="Rating",

                 yaxis_title="Count")

fig.show()
fig = px.histogram(df, x="course_rating", color='course_Certificate_type')

fig.update_layout(title_text="Rating distribution each certification",

                 xaxis_title="Rating",

                 yaxis_title="Count")

fig.show()
course_df = pd.DataFrame(df.groupby('course_title')['course_rating'].mean())



course_df = course_df.sort_values(by='course_rating', ascending=False)[:20]



course_df = course_df.reset_index()
course_df
level_df = pd.DataFrame(df.groupby('course_difficulty')['course_students_enrolled'].sum())



level_df = level_df.reset_index()
fig = px.bar(level_df.sort_values(by='course_students_enrolled',ascending=False), x="course_difficulty", y="course_students_enrolled", color='course_difficulty')

fig.update_layout(title_text="Number of students each course level",

                 xaxis_title="Level",

                 yaxis_title="Count")

fig.show()
course_enroll_df = pd.DataFrame(df.groupby('course_title')['course_students_enrolled'].sum())



course_enroll_df = course_enroll_df.reset_index()



course_enroll_df = course_enroll_df.sort_values(by='course_students_enrolled',ascending=False)
fig = px.bar(course_enroll_df.sort_values(by='course_students_enrolled',ascending=False)[:20], x="course_title", y="course_students_enrolled", color='course_title')

fig.update_layout(title_text="10 most popular course title",

                 xaxis_title="Course title",

                 yaxis_title="Count")

fig.show()
fig = px.scatter(df, x="course_students_enrolled", y="course_rating")

fig.update_layout(title="Number of students and course ratings",

                 xaxis_title="Number of students",

                 yaxis_title="Rating"

                 )

fig.show()
fig = px.scatter(df, x="course_rating", y="course_students_enrolled")

fig.update_layout(title="Course ratings and number of students",

                 xaxis_title="Rating",

                 yaxis_title="Number of students"

                 )

fig.show()