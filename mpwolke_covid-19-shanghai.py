#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT72jGRpDdXhCyNI7k28qbxadRkMeKMMC0-5wZjblpDj35loLuX',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

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
df = pd.read_excel('/kaggle/input/2019ncovshanghaichina/2019-nCoV-Shanghai-China.xls')

df.head()
df.dtypes
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        colormap='Set3',

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(df['Description'])
cnt_srs = df['Description'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Description distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Description")
fig = px.pie( values=df.groupby(['Description']).size().values,names=df.groupby(['Description']).size().index)

fig.update_layout(

    title = "Description",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
fig = px.histogram(df[df.Feature.notna()],x="Feature",marginal="box",nbins=10)

fig.update_layout(

    title = "Feature",

    xaxis_title="Feature",

    yaxis_title="Number of features",

    barmode="group",

    bargap=0.1,

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )

py.iplot(fig)
fig = px.pie( values=df.groupby(['FeatureCN']).size().values,names=df.groupby(['FeatureCN']).size().index)

fig.update_layout(

    title = "FeatureCN  distribution ",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
df_aux = df[df.Feature.notna()]

df_aux=df_aux[df_aux.Description.notna()]

#df_patients_aux=df_patients_aux.Description.notna()

fig = px.histogram(df_aux,x="Feature",color="Description",marginal="box",opacity=1,nbins=10)

fig.update_layout(

    title = "Cases by Feature and Description",

    xaxis_title="Feature",

    yaxis_title="Description of cases",

    barmode="group",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    ))

py.iplot(fig)
sns.countplot(df["FeatureCN"])

plt.xticks(rotation=90)

plt.show()
sns.countplot(df["Sheet"])

plt.xticks(rotation=90)

plt.show()
sns.countplot(df["Feature"])

plt.xticks(rotation=90)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQLl5UpeS_Dwb0cMJHsOvkyHwo6otjI6rH-xGMtLUgg3DdB1pNE',width=400,height=400)