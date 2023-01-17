#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

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
df = pd.read_csv('../input/stack-overflow-tag-network/stack_network_nodes.csv', encoding='ISO-8859-2')

df.head()
fig = px.bar(df, x= "name", y= "nodesize", color_discrete_sequence=['crimson'],)

fig.show()
fig = px.line(df, x="name", y="nodesize", color_discrete_sequence=['green'], 

              title="StackOverflow")

fig.show()
fig = px.scatter(df, x="name", y="nodesize", color="group",

                 color_continuous_scale=["red", "green", "blue"])



fig.show()
fig = px.pie(df, values=df['group'], names=df['name'],

             title='Stack Overflow',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
df1 = pd.read_csv('../input/stack-overflow-tag-network/stack_network_links.csv', encoding='ISO-8859-2')

df1.head()
fig = px.bar(df1, x= "source", y= "value", color_discrete_sequence=['lawngreen'],)

fig.show()
fig = px.line(df1, x="source", y="value", color_discrete_sequence=['indigo'], 

              title="StackOverflow")

fig.show()
fig = px.scatter(df1, x="source", y="value", color="target",

                 color_continuous_scale=["Cividis"])



fig.show()
fig = px.pie(df1, values=df1['value'], names=df1['source'],

             title='Stack Overflow',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df1.source)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="green").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTwF_SiSxIh16E17T9oW5IhDpdlcFeWt16Oy6nDemrEWYfgGXEh&usqp=CAU',width=400,height=400)