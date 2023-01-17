#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTG9ZrPu_mogUIJxwQaS-O5sWU2PgVvVVtWmS7plowGZfvCSE88&usqp=CAU',width=400,height=400)
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
df = pd.read_csv("../input/population-per-municipality-in-rs-2008-census/opstine.csv")

df.head()
px.histogram(df, x='1690433', color='БЕОГРАД')
fig = px.bar(df, 

             x='БЕОГРАД', y='1690433', color_discrete_sequence=['#D63230'],

             title='Census 2008 - БЕОГРАД ', text='БЕОГРАД')

fig.show()
seaborn.set(rc={'axes.facecolor':'3F3FBF', 'figure.facecolor':'3F3FBF'})

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("БЕОГРАД - Census 2008")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df.index, y=df['1690433'])



# Add label for vertical axis

plt.ylabel("БЕОГРАД - Census 2008")
fig = px.line(df, x="БЕОГРАД", y="1690433", 

              title="БЕОГРАД - Census 2008")

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.БЕОГРАД)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()