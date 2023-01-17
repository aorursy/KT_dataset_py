import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/videoactivityrecognitionpatents/video_activity_recognition_patents.csv', engine='python')

df.head()
df['Applicants'].value_counts().to_frame().head(20)
top30ipc = pd.Series([code for line in df['IPC'].tolist() for code in line.split( '\n')]).value_counts().to_frame().reset_index(level=0).iloc[:30, :]

top30ipc.columns = ['IPC', 'counts']

top30ipc.head()
import plotly.express as px

fig = px.bar(top30ipc, x='IPC', y='counts', color=top30ipc.index)

fig.show()
top30cpc = pd.Series([code for line in df['CPC'].dropna().tolist() for code in line.split( '\n')]).value_counts().to_frame().reset_index(level=0).iloc[:30, :]

top30cpc.columns = ['CPC', 'counts']

top30cpc.head()
import plotly.express as px

fig = px.bar(top30cpc, x='CPC', y='counts', color=top30cpc.index)

fig.show()
pubdates = df['Earliest priority'].apply(lambda x: x.split('\n')[0]).to_frame()

px.scatter(pubdates)