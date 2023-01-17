#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQj8FmaHLTTVhLOn9MTbhCUPWvqOGE1S4spib9x-ImQrWR2pngq&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQAZBWNbLUzbgEE388RQ_xWjA_WG6dFvUWvmRMmA_VynJaBWyeP&usqp=CAU',width=400,height=400)
df = pd.read_csv("../input/uncover/world_bank/specialist-surgical-workforce-per-100-000-population.csv")

df.head()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
import plotly.express as px



# Grouping it by job title and country (Andre Sionek's code Kaggle Survey)

plot_data = df.groupby(['country_name', '2014'], as_index=False).indicator_name.sum()



fig = px.bar(plot_data, x='country_name', y='indicator_name', color='2014')

fig.show()
plot_data = df.groupby(['country_name'], as_index=False).indicator_code.sum()



fig = px.line(plot_data, x='country_name', y='indicator_code')

fig.show()
plot_data = df.groupby(['2014'], as_index=False).indicator_name.sum()



fig = px.line(plot_data, x='2014', y='indicator_name')

fig.show()
fig = px.scatter(df, x= "country_name", y= "indicator_name")

fig.show()
fig = px.bar(df, x= "country_name", y= "2014")

fig.show()
fig = px.scatter(df, x= "2014", y= "indicator_name")

fig.show()
fig = px.bar(df, x= "indicator_name", y= "2014")

fig.show()
cnt_srs = df['indicator_name'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='Countries',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="indicator_name")
cnt_srs = df['2014'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='2014 year',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="2014")
df1 = pd.read_csv("../input/uncover/world_bank/physicians-per-1-000-people.csv")

df1.head()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df1.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
sns.countplot(df["2017"])

plt.xticks(rotation=90)

plt.show()
fig = px.bar(df, x= "country_name", y= "1990")

fig.show()
df2 = pd.read_csv("../input/doctors-and-nurses-per-1000-people-by-country/Nurses_Per_Capital_By_Country.csv")

df2.head()
df2["TIME"].plot.hist()

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df2.INDICATOR)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
midvives = pd.read_csv("../input/uncover/world_bank/nurses-and-midwives-per-1-000-people.csv")

midvives.head()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in midvives.indicator_name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="orange").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
doctors = pd.read_csv("../input/doctors-and-nurses-per-1000-people-by-country/Doctors_Per_Capital_By_Country.csv")

doctors.head()
def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in doctors["INDICATOR"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='green', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Thank You Health Professionals')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRcLNS_HPm0YlinuQ1z5PMoTfwAcVjy30QWWJLCgcSc-TGWgTUg&usqp=CAU',width=400,height=400)