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

df = pd.read_csv('../input/cusersmarildownloadshostcsv/host.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'host.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
fig = px.bar(df, 

             x='Sex', y='Time residing in Union', color_discrete_sequence=['#27F1E7'],

             title='Time residing in Union by Gender', text='Upazila of residence')

fig.show()
fig = px.bar(df, 

             x='Sex', y='Upazila of residence', color_discrete_sequence=['crimson'],

             title='Upazila of Residence by Gender', text='Time residing in Union')

fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('Have you ever helped a Rohingya (e.g. financially, water supply, training)?').size()/df['Sex'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('How often do you interact with the Rohingya (e.g. exchange conversation, buy products from Rohingya, work with Rohingya)? ').size()/df['Sex'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('How well do you think you get on (communicate in general) with the Rohingya? ').size()/df['Sex'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
sns.countplot(x="Have you noticed any changes to tourism in your union since the recent arrival of the Rohingya (since 2016)? ",data=df,palette="GnBu_d",edgecolor="black")

plt.title('Changes to Tourism', weight='bold')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
df.columns.tolist()
sns.countplot(x='Should the Rohingya be allowed access to the same facilities (e.g. schools, hospitals, mosques, community centres) and services as the locals?',data=df,palette="rainbow",edgecolor="black")

plt.title('Access to the Same Facilities', weight='bold')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
fig = px.bar(df[['Union of residence ','Do you believe that the Rohingya want to return to Myanmar? ']].sort_values('Do you believe that the Rohingya want to return to Myanmar? ', ascending=False), 

                        y = "Do you believe that the Rohingya want to return to Myanmar? ", x= "Union of residence ", color='Do you believe that the Rohingya want to return to Myanmar? ', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Do you believe that the Rohingya want to return to Myanmar?")



fig.show()
fig = px.bar(df[['Sex','How often do you interact with the Rohingya (e.g. exchange conversation, buy products from Rohingya, work with Rohingya)? ']].sort_values('How often do you interact with the Rohingya (e.g. exchange conversation, buy products from Rohingya, work with Rohingya)? ', ascending=False), 

                        y = "How often do you interact with the Rohingya (e.g. exchange conversation, buy products from Rohingya, work with Rohingya)? ", x= "Sex", color='How often do you interact with the Rohingya (e.g. exchange conversation, buy products from Rohingya, work with Rohingya)? ', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='darkgreen', size=14))

fig.update_layout(title_text="How often do you interact with the Rohingya (e.g. exchange conversation, buy products from Rohingya, work with Rohingya)?")



fig.show()
fig = px.pie(df, values=df['ID'], names=df['Sex'],

             title='Bangladeshi Survey Participants by Gender',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('What safety concerns do you and your family experience in your community, if any? (maximum 3 responses) ').size()/df['Union of residence '].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
import plotly.express as px



# Grouping it by sex and question

plot_data = df.groupby(['Union of residence ', 'Do you think the Rohingya can make positive contributions to your community? '], as_index=False).ID.sum()



fig = px.bar(plot_data, x='Union of residence ', y='ID', color='Do you think the Rohingya can make positive contributions to your community? ', title='Do you think the Rohingya can make positive contributions to your community?')

fig.show()
import plotly.express as px



# Grouping it by sex and question

plot_data = df.groupby(['Union of residence ', 'Have you noticed any changes in your family\x92s income since the recent arrivals of the Rohingya (since 2016)?'], as_index=False).ID.sum()



fig = px.bar(plot_data, x='Union of residence ', y='ID', color='Have you noticed any changes in your family\x92s income since the recent arrivals of the Rohingya (since 2016)?', title='Any changes in family’s income since the arrivals of the Rohingya (since 2016)?')

fig.show()
import plotly.express as px



# Grouping it by sex and question

plot_data = df.groupby(['Upazila of residence', 'Have you noticed any changes in the prices of goods (e.g. vegetables, fruit, clothing, meat, fish, wood) since the recent arrivals of the Rohingya (since 2016)? '], as_index=False).ID.sum()



fig = px.bar(plot_data, x='Upazila of residence', y='ID', color='Have you noticed any changes in the prices of goods (e.g. vegetables, fruit, clothing, meat, fish, wood) since the recent arrivals of the Rohingya (since 2016)? ', title='Any changes in the prices of goods (e.g. fruit, clothing, meat) since the arrivals of the Rohingya, 2016?')

fig.show()
import plotly.express as px



# Grouping it by sex and question

plot_data = df.groupby(['Union of residence ', 'Have you noticed any changes in the prices of goods (e.g. vegetables, fruit, clothing, meat, fish, wood) since the recent arrivals of the Rohingya (since 2016)? '], as_index=False).ID.sum()



fig = px.bar(plot_data, x='Union of residence ', y='ID', color='Have you noticed any changes in the prices of goods (e.g. vegetables, fruit, clothing, meat, fish, wood) since the recent arrivals of the Rohingya (since 2016)? ', title='Any changes in the prices of goods (e.g. fruit, clothing, meat) since the arrivals of the Rohingya, 2016?')

fig.show()
ax = df['Do you think the Rohingya can make positive contributions to your community? '].value_counts().plot.barh(figsize=(10, 4))

ax.set_title('Can Rohingya make positive contributions to your community?', size=18)

ax.set_ylabel('Can Rohingya make positive contributions to your community?', size=14)

ax.set_xlabel('Count', size=12)
ax = df['Do you feel the Rohingya could eventually integrate and stay in Bangladesh indefinitely? '].value_counts().plot.barh(figsize=(10, 4), color='r')

ax.set_title('Could The Rohingya eventually stay in Bangladesh indefinitely?', size=18)

ax.set_ylabel('Could Rohingya eventually stay in Bangladesh indefinitely?', size=14)

ax.set_xlabel('Count', size=12)
ax = df['Do you believe that the Rohingya will be eventually repatriated (in the next two years)? '].value_counts().plot.barh(figsize=(10, 4), color='purple')

ax.set_title('Would The Rohingya be repatriated in the next two years?', size=18)

ax.set_ylabel('Would Rohingya be repatriated in the next two years?', size=14)

ax.set_xlabel('Count', size=12)
import plotly.express as px



# Grouping it by sex and question

plot_data = df.groupby(['Union of residence ', 'Do you feel positive about your and your family\x92s future her in Bangladesh?'], as_index=False).ID.sum()



fig = px.bar(plot_data, x='Union of residence ', y='ID', color='Do you feel positive about your and your family\x92s future her in Bangladesh?', title='')

fig.show()