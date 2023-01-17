# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import seaborn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/medical-translation/medi_translate.csv")

df.head()
fig = px.bar(df, 

             x='Translation', y='Language', color_discrete_sequence=['#27F1E7'],

             title='Medical Translation', text='Translation')

fig.show()
fig = px.bar(df, 

             x='Text', y='Category', color_discrete_sequence=['#7F3FBF'],

             title='Medical Translation', text='Text')

fig.show()
px.histogram(df, x='Text', color='Category')
seaborn.set(rc={'axes.facecolor':'magenta', 'figure.facecolor':'magenta'})

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Medical Translation")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df.index, y=df['Category'])



# Add label for vertical axis

plt.ylabel("Medical Translation")
seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'#27F1E7'})

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Medical Translation")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df.index, y=df['Language'])



# Add label for vertical axis

plt.ylabel("Medical Translation")
fig = px.line(df, x="Text", y="Category", 

              title="Medical Translation")

fig.show()
fig = px.bar(df, x= "Translation", y= "Language")

fig.show()
seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'#27F1E7'})

sns.countplot(df["Language"])

plt.xticks(rotation=90)

plt.show()
from wordcloud import WordCloud

def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in df["Language"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Medical Translation')