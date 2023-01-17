# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/ufo-sightings/ufo.csv', encoding='ISO-8859-2')

df.head()
fig = px.bar(df, 

             x='Shape Reported', y='Colors Reported', color_discrete_sequence=['#27F1E7'],

             title='UFO Sightings', text='Time')

fig.show()
fig = px.bar(df, 

             x='Colors Reported', y='Shape Reported', color_discrete_sequence=['crimson'],

             title='UFO Sightings', text='State')

fig.show()
fig = px.density_contour(df, x="Shape Reported", y="Colors Reported",title='UFO Sightings', color_discrete_sequence=['purple'])

fig.show()
fig = px.line(df, x="Shape Reported", y="Colors Reported", color_discrete_sequence=['darkseagreen'], 

              title="UFO Sightings")

fig.show()
fig = px.line(df, x="Time", y="Colors Reported", color_discrete_sequence=['orange'], 

              title="UFO Sightings")

fig.show()
fig = px.line(df, x="Time", y="Shape Reported", color_discrete_sequence=['magenta'], 

              title="UFO Sightings")

fig.show()
#Code from Gabriel Preda

def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

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
plot_count("Shape Reported", "Shape Reported", df,4)
plot_count("Colors Reported", "Colors Reported", df,4)
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'Shape Reported', data = df, palette="cool",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(132)

sns.countplot(x= 'Colors Reported', data = df, palette="flag",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'State', data = df, palette="Greens_r",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
fig = go.Figure(data=[go.Scatter(

    x=df['Time'][0:10],

    y=df['Shape Reported'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='UFO Sightings',

    xaxis_title="Time",

    yaxis_title="Shape Reported",

)

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.City)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
fig = go.Figure(data=[go.Bar(

            x=df['Time'][0:10], y=df['Shape Reported'][0:10],

            text=df['Shape Reported'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='UFO Sightings',

    xaxis_title="Time",

    yaxis_title="Shape Reported",

)

fig.show()