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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots
df = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

df.head()
sns.pairplot(df,hue='DEATH_EVENT')
maleage = df[df['sex']==1]['age']

femaleage = df[df['sex']==0]['age']

hist_data = [maleage,femaleage]



group_labels = ['Age Distribution for Male',"Age Distribution for Female"]

colors=['royalblue',"pink"]

fig = ff.create_distplot(hist_data, group_labels,colors=colors) #custom bin_size

fig.update_layout(title="Age Distribution by Gender",title_x=0.5)

fig.show()
df.head()
df['DEATH_EVENT'] = df['DEATH_EVENT'].astype(str)

df['sex'] = df['sex'].astype(str)

df['anaemia'] = df['anaemia'].astype(str)

df['diabetes'] = df['diabetes'].astype(str)

df['smoking'] = df['smoking'].astype(str)
fig = px.sunburst(df, path=['sex','DEATH_EVENT','smoking'], values='platelets', color='DEATH_EVENT')

fig.update_layout(title="Distribution by Sex, DEATH_EVENT, smoking",title_x=0.5)

fig.show()
fig = px.density_heatmap(df, x="age", y="platelets", facet_row="DEATH_EVENT", facet_col="sex")

fig.update_layout(title='Density heatmap of Age vs Platelets with Gender and DEATH_EVENT')

fig.show()
fig = px.parallel_categories(df, dimensions=['sex','DEATH_EVENT','smoking'],

                color="age", color_continuous_scale=px.colors.sequential.Inferno) # labeling

fig.update_layout(title="Heart Data Parallel Categories Diagram ",title_x=0.5)

fig.show()
df.head()
fig = px.scatter_3d(df, x='age', y='ejection_fraction', z='platelets',

              color='sex', size_max=18,

              symbol='sex', opacity=0.7)



fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig = px.scatter_3d(df, x='age', y='ejection_fraction', z='platelets',

              color='DEATH_EVENT', size_max=18,

              symbol='DEATH_EVENT', opacity=0.7)



fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,cmap='viridis')

plt.show()
net_category=df['diabetes'].value_counts().to_frame().reset_index().rename(columns={'index':'diabetes','diabetes':'count'})

colors=['yellow','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['diabetes'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="diabetes Distribution",title_x=0.5)

fig.show()
net_category=df['high_blood_pressure'].value_counts().to_frame().reset_index().rename(columns={'index':'high_blood_pressure','high_blood_pressure':'count'})

colors=['yellow','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['high_blood_pressure'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="high_blood_pressure Distribution",title_x=0.5)

fig.show()
net_category=df['anaemia'].value_counts().to_frame().reset_index().rename(columns={'index':'anaemia','anaemia':'count'})

colors=['yellow','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['anaemia'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="anaemia Distribution",title_x=0.5)

fig.show()
net_category=df['smoking'].value_counts().to_frame().reset_index().rename(columns={'index':'smoking','smoking':'count'})

colors=['yellow','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['smoking'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="smoking Distribution",title_x=0.5)

fig.show()
net_category=df['sex'].value_counts().to_frame().reset_index().rename(columns={'index':'sex','sex':'count'})

colors=['royalblue','pink']

fig = go.Figure([go.Pie(labels=net_category['sex'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="Gender Distribution",title_x=0.5)

fig.show()
df.head()
plt.figure(figsize=(10,8))

sns.distplot(df['creatinine_phosphokinase'],color='red')

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['platelets'],color='red')

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['serum_sodium'],color='red')

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['serum_creatinine'],color='red')

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['ejection_fraction'],color='red')

plt.show()