import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')
df.shape
df.head(4)
df.isna().sum()
df.info(memory_usage='deep')
df.course_Certificate_type.value_counts().plot(kind='bar')
df.course_difficulty.value_counts().plot(kind='bar')
df.course_students_enrolled = df.course_students_enrolled.apply(lambda x : float(str(x).replace('k', '').replace('m',''))*1000)
top_10 = df.nlargest(10, 'course_students_enrolled')
fig = px.bar(top_10, x='course_title', y='course_students_enrolled')

fig.update_layout(

    title = 'Top 10 courses by number of students enrolled',

    xaxis_title="Courses",

    yaxis_title="Students enrolled",

)

fig.show()
course_dif = df.groupby('course_difficulty')['course_students_enrolled'].sum().reset_index()

fig = px.bar(course_dif.sort_values(by = 'course_students_enrolled', ascending=False), x='course_difficulty', y='course_students_enrolled')

fig.update_layout(

    title = 'Top 10 courses by number of students enrolled',

    xaxis_title="Courses",

    yaxis_title="Students enrolled",

)

fig.show()