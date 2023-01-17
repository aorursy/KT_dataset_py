



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/students-performance/StudentsPerformance.csv')

df.head(5)
df.info()
df.columns
df.columns = ['gender', 'race', 'parent_edu', 'lunch', 'test_prep', 'math_s', 'reading_s', 'writing_s']

df.columns
df['gender'].value_counts()
df['total'] = df.math_s + df.reading_s + df.writing_s

df.head(5)


fig = px.histogram(df, x="total", color="gender")

fig.show()


fig = px.box(df, x="race", y="total", color="gender")

fig.update_traces(quartilemethod="exclusive") 

fig.show()


fig = px.box(df, x="parent_edu", y="total", color="gender")

fig.update_traces(quartilemethod="exclusive") 

fig.show()
fig = px.violin(df, y="total", x="lunch", color="gender", box=True, points="all",

          hover_data=df.columns)

fig.show()
fig = px.violin(df, y="total", x="test_prep", color="gender", box=True, points="all",

          hover_data=df.columns)

fig.show()