#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
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
df = pd.read_csv("../input/externalourworldindata/share-of-population-urban.csv")
df.head().style.background_gradient(cmap='nipy_spectral')
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='Reds')

plt.show()
df.dropna(how = 'all',inplace = True)

df.drop(['Code'],axis=1,inplace = True)

df.shape
plot_data = df.groupby(['Year'], as_index=False).Entity.sum()



fig = px.line(plot_data, x='Year', y='Entity')

fig.show()
df = df.rename(columns={'Urban population (% of total) (% of total)':'urbanpopulation' })
plot_data = df.groupby(['Year'], as_index=False).urbanpopulation.sum()



fig = px.line(plot_data, x='Year', y='urbanpopulation')

fig.show()
cnt_srs = df['Year'].value_counts().head()

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

    title='Urban Population by Year',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Year")
cnt_srs = df['Entity'].value_counts().head()

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

    title='Entity',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Entity")
# Grouping it by age and province

plot_data = df.groupby(['Year', 'Entity'], as_index=False).urbanpopulation.sum()



fig = px.bar(plot_data, x='Year', y='urbanpopulation', color='Entity')

fig.show()
dfcorr=df.corr()

dfcorr
plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=False,cmap='vlag')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSfZHGEaUNxPCCSUd-iv3lCLVvD3rqfDKsP8zzjfB3ik2EMbaKv&usqp=CAU',width=400,height=400)