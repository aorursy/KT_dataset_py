#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTQNy7iKEc7_VMYYpRiW4oRjFMX0rzy-2nkAD6u9Btdkf8jC6ku&usqp=CAU',width=400,height=400)
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
df = pd.read_csv("../input/genome-information-by-organism/plasmids.csv")

df.head().style.background_gradient(cmap='nipy_spectral')
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.dropna(how = 'all',inplace = True)

df.drop(['Strain', 'BioSample'],axis=1,inplace = True)

df.shape
genome = df.groupby(["CDS", "Neighbors"])["Genes"].sum().reset_index().sort_values("Genes",ascending=False).reset_index(drop=True)

genome
fig = go.Figure(data=[go.Bar(

            x=genome['Genes'][0:10], y=genome['CDS'][0:10],

            text=genome['Genes'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Plasmids Genome',

    xaxis_title="Genes",

    yaxis_title="CDS",

)

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=genome['Genes'][0:10],

    y=genome['CDS'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Plasmids Genome',

    xaxis_title="Genes",

    yaxis_title="CDS",

)

fig.show()
labels = ["CDS","Genes"]

values = genome.loc[0, ["CDS","Genes"]]

df = px.data.tips()

fig = px.pie(genome, values=values, names=labels, color_discrete_sequence=['royalblue','darkblue','lightcyan'])

fig.update_layout(

    title='Plasmid Genome : '+str(genome["Genes"][0]),

)

fig.show()
fig = px.pie(genome, values=genome['Genes'], names=genome['CDS'],

             title='Plasmids Genome',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTTSyvQdbmSFK1P0b4B_yr9Evf5012fuHuf-Eml0ly7wr0irGJx&usqp=CAU',width=400,height=400)