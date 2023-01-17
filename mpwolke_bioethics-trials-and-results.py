# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import plotly.graph_objects as go

import plotly.offline as py



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsbioethicscsv/bioethics.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'bioethics.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
df.isnull().sum()
cnt_srs = df['Study_Population'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Study Population Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Study_Population")
cnt_srs = df['Study_Designs'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='Study Designs Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Study_Designs")
# Count Plot

plt.style.use("classic")

plt.figure(figsize=(10, 8))

sns.countplot(df['Study_Population'], palette='Accent_r')

plt.xlabel("Study Population")

plt.ylabel("Count")

plt.title("Study Population")

plt.xticks(rotation=45, fontsize=8)

plt.show()
sns.countplot(x="Participants",data=df,palette="GnBu_d",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
#Code from Gabriel Preda

#plt.style.use('dark_background')

def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set2')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("Study_Population", "Study Population", df,4)
plot_count("Study_Designs", "Study_Designs", df,4)
ax = df['Study_Population'].value_counts().plot.barh(figsize=(14, 6))

ax.set_title('Study Population Distribution', size=18)

ax.set_ylabel('Study Population', size=14)

ax.set_xlabel('Count', size=14)
ax = df['Study_Designs'].value_counts().plot.barh(figsize=(14, 6), color='r')

ax.set_title('Study Designs Distribution', size=18, color='r')

ax.set_ylabel('Study Designs', size=14)

ax.set_xlabel('Count', size=14)
ax = df['Phase'].value_counts().plot.barh(figsize=(14, 6), color='g')

ax.set_title('Phase Distribution', size=18, color='g')

ax.set_ylabel('Study Phase', size=14)

ax.set_xlabel('Count', size=14)
fig = px.bar(df[['Study_Population','Participants']].sort_values('Participants', ascending=False), 

                        y = "Participants", x= "Study_Population", color='Participants', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Bioethics Trials and Results")



fig.show()
sns.countplot(df['Results_First_Received'],linewidth=3,palette="Set2",edgecolor='black')

plt.xticks(rotation=45)

plt.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('Study_Population').size()/df['Participants'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values,marker_colors = px.colors.sequential.speed, hole=.6)])

fig.show()
plt.figure(figsize=(18,6))

plt.subplot(1, 2, 1)

sns.countplot(x=df['Study_Population'],hue=df['Results_First_Received'],palette='summer',linewidth=3,edgecolor='white')

plt.xticks(rotation=45)

plt.title('First Received Results')

plt.subplot(1, 2, 2)

sns.countplot(x=df['Conditions'],hue=df['Primary_Completion_Date'],palette='hot',linewidth=3,edgecolor='white')

plt.xticks(rotation=45)

plt.title('Primary Completion Dates')

plt.show()
fig = px.bar(df, x= "First_Received", y= "Conditions", color_discrete_sequence=['crimson'], title='First Received & Participants Conditions')

fig.show()
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'Certificate_of_Delay', data = df, palette="GnBu_d",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(132)

sns.countplot(x= 'US_Clinical_Trial_Site', data = df, palette="flag",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'Publication_Date', data = df, palette="Greens_r",edgecolor="black")

plt.xticks(rotation=45)

plt.show()