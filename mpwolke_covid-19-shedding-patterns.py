#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
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

import seaborn

from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/aipowered-literature-review-csvs/kaggle/working/TIE/What do we know about viral shedding in stool_.csv', encoding='ISO-8859-2')

df.head()
df = df.rename(columns={'Unnamed: 0':'unnamed', 'Study Type': 'study', 'Source / Summary': 'summary', 'Sample Size': 'size', 'Days After Onset': 'onset' })
fig = px.bar(df,

             y='size',

             x='unnamed',

             orientation='h',

             color='size',

             title='Viral Shedding Patterns in Stool',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.area(df,

            x='size',

            y='unnamed',

            template='plotly_dark',

            color_discrete_sequence=['rgb(18, 115, 117)'],

            title='Viral Shedding Patterns in Stool',

           )



fig.update_yaxes(range=[0,2])

fig.show()
fig = px.bar(df, 

             x='onset', y='unnamed', color_discrete_sequence=['#27F1E7'],

             title='Viral Shedding Patterns in Stool', text='size')

fig.show()
fig = px.bar(df, 

             x='Confirmation', y='study', color_discrete_sequence=['crimson'],

             title='Viral Shedding Patterns in Stool', text='Confirmation')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41591-020-0817-4/MediaObjects/41591_2020_817_Fig1_HTML.png?as=webp',width=400,height=400)
fig = px.density_contour(df, x="study", y="unnamed", color_discrete_sequence=['purple'])

fig.show()
fig = px.line(df, x="Date", y="size", color_discrete_sequence=['darkseagreen'], 

              title="Viral Sheddings Patterns in Stool")

fig.show()
df1 = pd.read_csv('../input/aipowered-literature-review-csvs/kaggle/working/TIE/What do we know about viral shedding in urine_.csv', encoding='ISO-8859-2')

df1.head()
fig = px.bar(df1,

             y='Sample Size',

             x='Unnamed: 0',

             orientation='h',

             color='Sample Size',

             title='Viral Shedding Patterns in Urine',

             opacity=0.8,

             color_discrete_sequence=px.colors.sequential.Bluyl,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.area(df1,

            x='Sample Size',

            y='Unnamed: 0',

            template='plotly_dark',

            color_discrete_sequence=['rgb(220, 248, 165)'],

            title='Viral Shedding Patterns in Urine',

           )



fig.update_yaxes(range=[0,2])

fig.show()
fig = px.bar(df1, 

             x='Date', y='Sample Size', color_discrete_sequence=['#27F1E7'],

             title='Viral Shedding Patterns in Urine', text='Date')

fig.show()
fig = px.histogram(df1[df1.Date.notna()],x="Date",marginal="box",nbins=10)

fig.update_layout(

    title = "Viral Sheddings Patterns in Urine",

    xaxis_title="Date",

    yaxis_title="Unnamed: 0",

    template='plotly_dark',

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
df2 = pd.read_csv('../input/aipowered-literature-review-csvs/kaggle/working/TIE/What do we know about viral shedding in blood_.csv', encoding='ISO-8859-2')

df2.head()
fig = px.bar(df2,

             y='Sample Size',

             x='Unnamed: 0',

             orientation='h',

             color='Sample Size',

             title='Viral Shedding Patterns in Blood',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Spectral,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.area(df2,

            x='Sample Size',

            y='Unnamed: 0',

            template='plotly_dark',

            color_discrete_sequence=['rgb(243, 25, 37)'],

            title='Viral Shedding Patterns in Blood',

           )



fig.update_yaxes(range=[0,2])

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQyQmJyY9aoj4iVos3rET-saSdhBu6AuE1XdQaOPPuVilYiS4QB&usqp=CAU',width=400,height=400)