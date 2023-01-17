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
df = pd.read_csv("../input/medical-care-questions/covid19_medical_care_questions.csv")

df.head()
seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'#27F1E7'})

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Medical Care Questions")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df.index, y=df['question'])



# Add label for vertical axis

plt.ylabel("Medical Care Questions")
seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'#27F1E7'})

sns.countplot(df["question"])

plt.xticks(rotation=90)

plt.show()
cnt_srs = df['question'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'haline',

        reversescale = True

    ),

)



layout = dict(

    title='Medical Care Questions',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="question")
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.question)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()