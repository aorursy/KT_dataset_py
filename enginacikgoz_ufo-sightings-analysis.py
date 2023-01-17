# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True) 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load data and change theme.

df = pd.read_csv("../input/scrubbed.csv", low_memory=False)

plt.style.use('dark_background') # I love dark theme. :)



df.info()
df.country.unique()

df["country"] = df["country"].fillna("?")

df.country.unique()

df.sample(5)
def grep_year(x):

    x = x.split(" ")[0]

    x = x.split("/")[2]

    x = int(x)

    return x



df["Sight-Year"] = df['datetime'].apply(grep_year)

df["Date-Posted-Year"] = df['date posted'].apply(grep_year)
df.sample(5)
def conv_season(x):

    x = int(x.split("/")[0])

    

    if x in range(3,6):

        return "Spring"

    if x in range(6,9):

        return "Summer"

    if x in range(9,12):

        return "Autumn"

    if x == 12 or x == 1 or x == 2:

        return "Winter"



df["Season"] = df['datetime'].apply(conv_season)
df.sample(5)
states_us = df[df.country == "us"]["state"].value_counts().index

states_ratio = df[df.country == "us"]["state"].value_counts().values

states_us = [i.upper() for i in states_us]



plt.subplots(figsize=(24,8))

sns.barplot(states_us, states_ratio)

plt.xticks(rotation=45, fontsize=16)

plt.yticks(fontsize=20)



plt.show()
plt.subplots(figsize=(22,8))

duration_sec = [i for i in df["duration (seconds)"].value_counts()]

duration_sec_list = []

for i in duration_sec:

    if i in range(0,16):

        duration_sec_list.append("0-15")

    if i in range(15,31):

        duration_sec_list.append("15-30")

    if i in range(31,61):

        duration_sec_list.append("30-60")

    if i in range(60,121):

        duration_sec_list.append("60-120")

    if i in range(120,241):

        duration_sec_list.append("120-240")

    if i > 240:

        duration_sec_list.append(">240")

duration_sec_list = pd.Series(duration_sec_list)

di = duration_sec_list.value_counts().index

dv = duration_sec_list.value_counts().values

sns.barplot(di,dv)



plt.xlabel("Time - Seconds",fontsize=24)

plt.xticks(fontsize=20)

plt.ylabel("Rates",fontsize=24)

plt.yticks(fontsize=20)



plt.show()
plt.subplots(figsize=(18,8))



df['shape'].value_counts().plot('bar')

plt.xticks(rotation=45, fontsize=15)

plt.show()
plt.subplots(figsize=(14,6))



df['Season'].value_counts().plot('bar')

plt.ylabel("Frequency")

plt.xticks(rotation=0)

plt.title("Sight - Season")

plt.show()
plt.subplots(figsize=(22,10))



plt.subplot(2,1,1)

plt.title("Sight rates by years")

df['Sight-Year'].value_counts().plot('bar')

plt.xlabel("Years")

plt.subplots(figsize=(22,10))



plt.subplot(2,1,2)

plt.title("Posting the case's rates by years")

df['Date-Posted-Year'].value_counts().plot('bar')

plt.ylabel("Post Year")

plt.xticks(rotation=0)

plt.show()

data = [

        dict(

        type='choropleth',

        locations = states_us,

        z = states_ratio,

        locationmode = 'USA-states',

        text = "times",

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Sight rates by states")

        )

        ]



layout = dict(

        title = 'UFO sight rates from USA',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

              )





fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
words = [i for i in df.comments.dropna()]

    

words = " ".join(words)



plt.subplots(figsize=(28,12))

wordcloud = WordCloud(

                          background_color='black',

                          width=2048,

                          height=1024

                          ).generate(words)

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()