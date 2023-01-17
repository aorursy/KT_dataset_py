#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT6XLL780qmRyGzYi2yJ0XLKpc8-adpYskc9ZuCZ5Qm2VJwHPnD&usqp=CAU',width=400,height=400)
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
df = pd.read_csv("../input/covid19-challenges/npi_canada.csv")

df.head()
df = df.rename(columns={"target_population_category":"population","enforcement_category":"enforcement", "oxford_government_response_category": "response"})
df_grp = df.groupby(["start_date","id"])[["population","enforcement","response", "region", "intervention_summary"]].sum().reset_index()

df_grp.head()
plt.style.use('dark_background')

plt.figure(figsize=(15, 5))

plt.title('ID')

df_grp.id.value_counts().plot.bar();
df_grp_plot = df_grp.tail(80)
fig=px.bar(df_grp_plot,x='id', y="population", animation_frame="start_date", 

           animation_group="id", color="id", hover_name="id")

fig.update_yaxes(range=[0, 1500])

fig.update_layout(title='Population and ID')
npi = df_grp.groupby(["id"])["population"].sum().reset_index().sort_values("id",ascending=False).reset_index(drop=True)

npi
fig = go.Figure(data=[go.Bar(

            x=npi['population'][0:10], y=npi['id'][0:10],

            text=npi['id'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='NonPharmaceutical Interventions -NPI',

    xaxis_title="Population",

    yaxis_title="ID",

)

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=npi['population'][0:10],

    y=npi['id'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='NonPharmaceutical Interventions - NPI',

    xaxis_title="Population",

    yaxis_title="ID",

)

fig.show()
#Code from Prashant Banerjee @Prashant111

plt.style.use('dark_background')

labels = npi['id'].value_counts().index

size = npi['id'].value_counts()

colors=['#EAEE10','#3FBF3F']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('id', fontsize = 20)

plt.legend()

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.response)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.intervention_summary)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="green").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.population)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="blue").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS8tg-E_6sGImSeNC7buXKfnKr3LD8jZV7CMqo53ZAjc63LSnto&usqp=CAU',width=400,height=400)