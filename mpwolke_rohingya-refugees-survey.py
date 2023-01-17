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

df = pd.read_csv('../input/cusersmarildownloadsrohingyacsv/rohingya.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'rohingya.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
df["Sex"].value_counts()
sns.countplot(x="Sex",data=df,palette="GnBu_d",edgecolor="black")

plt.title('Gender Distribution', weight='bold')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
df["If you could choose any country in the world, other than Myanmar, where would you go?"].value_counts()
sns.countplot(x="How do you feel about the future?",data=df,palette="rainbow",edgecolor="black")

plt.title('Expectations about the Future', weight='bold')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
df["How satisfied are you with the amount of space allocated to you and your family in the camp?"].value_counts()
sns.countplot(x="How satisfied are you with the amount of space allocated to you and your family in the camp?",data=df,palette="coolwarm",edgecolor="black")

plt.title('Satisfaction with amount of Space in the Camp', weight='bold')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
fig = px.bar(df[['Sex','How scared are you to leave for another country?']].sort_values('How scared are you to leave for another country?', ascending=False), 

                        y = "How scared are you to leave for another country?", x= "Sex", color='How scared are you to leave for another country?', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="How scared are you to leave for another country?")



fig.show()
fig = px.bar(df[['Sex','Do you think you are able to exercise your religion freely in the camp?']].sort_values('Do you think you are able to exercise your religion freely in the camp?', ascending=False), 

                        y = "Do you think you are able to exercise your religion freely in the camp?", x= "Sex", color='Do you think you are able to exercise your religion freely in the camp?', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=8))

fig.update_layout(title_text="Do you think you are able to exercise your religion freely in the camp?")



fig.show()
fig = px.pie(df, values=df['Row_ID'], names=df['Sex'],

             title='Rohingya Refugees by Gender',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('How satisfied are you with the quality of healthcare you and your family are receiving in the camp?').size()/df['Sex'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
import plotly.express as px



# Grouping it by sex and question

plot_data = df.groupby(['Sex', 'How do you mostly hear about the news?'], as_index=False).Row_ID.sum()



fig = px.bar(plot_data, x='Sex', y='Row_ID', color='How do you mostly hear about the news?', title='How do you mostly hear about the news?')

fig.show()
import plotly.express as px



# Grouping it by sex and question

plot_data = df.groupby(['Sex', 'Who do you consider the most important person in your camp?'], as_index=False).Row_ID.sum()



fig = px.bar(plot_data, x='Sex', y='Row_ID', color='Who do you consider the most important person in your camp?', title='Who do you consider the most important person in your camp?')

fig.show()
import plotly.express as px



# Grouping it by sex and question

plot_data = df.groupby(['Sex', 'Do you have enough good friends in the camp?'], as_index=False).Row_ID.sum()



fig = px.bar(plot_data, x='Sex', y='Row_ID', color='Do you have enough good friends in the camp?', title='Do you have enough good friends in the camp?')

fig.show()
ax = df['Do you believe Rohingya and Bangladeshis can be good friends?'].value_counts().plot.barh(figsize=(10, 4))

ax.set_title('Do you believe Rohingya and Bangladeshis can be good friends?', size=18)

ax.set_ylabel('Do you believe Rohingya and Bangladeshis can be good friends?', size=14)

ax.set_xlabel('Count', size=12)
ax = df['How much do you trust the NGOs in the camp?'].value_counts().plot.barh(figsize=(10, 4), color='r')

ax.set_title('How much do you trust the NGOs in the camp?', size=18)

ax.set_ylabel('How much do you trust the NGOs in the camp?', size=14)

ax.set_xlabel('Count', size=12)
ax = df['Do you often feel stressed or overwhelmed by your situation?'].value_counts().plot.barh(figsize=(10, 4), color='g')

ax.set_title('Do you often feel stressed or overwhelmed by your situation?', size=18)

ax.set_ylabel('Do you often feel stressed or overwhelmed by your situation?', size=14)

ax.set_xlabel('Count', size=12)
ax = df['Do you think that sexual harassment in the camp happens often?'].value_counts().plot.barh(figsize=(10, 4), color='orange')

ax.set_title('Do you think that sexual harassment in the camp happens often?', size=18)

ax.set_ylabel('Do you think that sexual harassment in the camp happens often?', size=14)

ax.set_xlabel('Count', size=12)
ax = df['Where does sexual harassment happen the most?'].value_counts().plot.barh(figsize=(10, 4), color='purple')

ax.set_title('Where does sexual harassment happen the most?', size=18)

ax.set_ylabel('Where does sexual harassment happen the most?', size=14)

ax.set_xlabel('Count', size=12)
df["How do you feel about the future?"].value_counts()
import plotly.express as px



# Grouping it by sex and question

plot_data = df.groupby(['Sex', 'How do you feel about the future?'], as_index=False).Row_ID.sum()



fig = px.bar(plot_data, x='Sex', y='Row_ID', color='How do you feel about the future?', title='How do you feel about the future?')

fig.show()