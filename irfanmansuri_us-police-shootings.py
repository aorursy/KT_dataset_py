! pip install dexplot
# Importing the necessary libraries

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objs as go
from plotly.offline import iplot
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
df = pd.read_csv('/kaggle/input/us-police-shootings/shootings.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
labels = df['state'].value_counts()[:10].index
values = df['state'].value_counts()[:10].values

colors = df['state']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    

import dexplot as dxp
dxp.count(val='state', data=df, figsize=(15,3), normalize=True)
dxp.count(val='state', data=df, figsize=(15,3), split='gender', normalize=True)
labels = df['manner_of_death'].value_counts()[:].index
values = df['manner_of_death'].value_counts()[:].values

colors = df['manner_of_death']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    

dxp.count(val='manner_of_death', data=df, figsize=(8,5), split='gender', normalize=True)
dxp.count(val='age', data=df, figsize=(20,5), normalize=True)
df['above40'] = ['above40'if i >=40 else 'below40'for i in df.age]

df['generation']="-"
for i in df.age.index:
    if df['age'][i] >=0 and df['age'][i]<40:
        df['generation'][i]='30s'
    elif df['age'][i] >=40 and df['age'][i]<50:
        df['generation'][i]='40s'
    elif df['age'][i] >=50 and df['age'][i]<60:
        df['generation'][i]='50s'
    else:
        df['generation'][i]='60+'
sns.countplot(x=df.above40,  palette="Set2")
plt.ylabel('Number of People', fontsize=10)
plt.title('Age of People', color='red', fontsize=15)
labels = df['race'].value_counts()[:].index
values = df['race'].value_counts()[:].values

colors = df['race']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    

dxp.count(val='race', data=df, figsize=(8,5), split='gender', normalize=True)
labels = df['signs_of_mental_illness'].value_counts()[:].index
values = df['signs_of_mental_illness'].value_counts()[:].values

colors = df['signs_of_mental_illness']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    

labels = df['threat_level'].value_counts()[:].index
values = df['threat_level'].value_counts()[:].values

colors = df['threat_level']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    

labels = df['flee'].value_counts()[:].index
values = df['flee'].value_counts()[:].values

colors = df['flee']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    

# Mental Illness and Fleeing

dxp.count(val='flee', data=df, figsize=(8,5), split='signs_of_mental_illness', normalize=True)
dxp.count(val='flee', data=df, figsize=(8,5), split='threat_level', normalize=True)
labels = df['body_camera'].value_counts()[:].index
values = df['body_camera'].value_counts()[:].values

colors = df['body_camera']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    

labels = df['arms_category'].value_counts()[:].index
values = df['arms_category'].value_counts()[:].values

colors = df['arms_category']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    

dxp.count(val='arms_category', data=df, figsize=(8,5), split='flee', normalize=True)
dxp.count(val='arms_category', data=df, figsize=(8,5), split='threat_level', normalize=True)
dxp.count(val='arms_category', data=df, figsize=(8,5), split='gender', normalize=True)
