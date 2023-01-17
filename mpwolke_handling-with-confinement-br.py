#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTRaDGg86JnCOMo40OoASs4fbgBq02KD30KaknycQy-sUsozOkR&usqp=CAU',width=400,height=400)
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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/coronavirus-rio-de-janeiro/coronavirus_rj_case_metadata.csv")

df.head().style.background_gradient(cmap='summer')
df = df.rename(columns={'bairro_resid__estadia':'bairro', 'ap__residencia': 'ap'})
dfcorr=df.corr()

dfcorr

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=False,cmap='summer')

plt.show()
corr = df.corr(method='pearson')

sns.heatmap(df.corr(), cmap='summer')
#code from Meenakshi https://www.kaggle.com/meenakshiramaswamy/covid-wk4-multiple-data-sources-xgboost

isol = df.groupby('dt_notific')['hospitalizações', 'óbitos', 'uti'].sum().reset_index()

isol = isol.melt(id_vars="dt_notific", value_vars=['hospitalizações', 'óbitos', 'uti'],

                 var_name='case', value_name='count')



fig = px.area(isol, x="dt_notific", y="count", color='case',

             title='Confinement in Rio de Janeiro', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()
fig = px.pie(df, values='objectid', names='bairro')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
fig = px.pie(df, values='ObjectId2', names='idade')

fig.update_traces(textposition='inside',textfont_size=14)

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
fig = px.bar(df, x= "dt_notific", y= "idade")

fig.show()
fig = px.scatter(df, x= "dt_notific", y= "idade")

fig.show()
#plt.style.use('dark_background')

fig=sns.lmplot(x="objectid", y="idade",data=df)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.bairro)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcShUkEcs6eCjrENAFYDRBtT4jrMrpRxSzM1n7FvZU3jHP2UNqbs&usqp=CAU',width=400,height=400)