#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQo-6L7Iax1FCrExDdm-CMNh-2Hqpp1WOcE7dREGSIrE9TK59DA&usqp=CAU',width=400,height=400)
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
df = pd.read_csv("../input/corona-turkiye/TRKYE CORONA VRS STATSTKLER.csv")

df.head().style.background_gradient(cmap='Reds_r')
df = df.rename(columns={'Günlük Vaka':'vaka', 'Günlük Ölüm': 'ölüm', 'Toplam Vakalar': 'vakalar'})
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='Reds')

plt.show()
df.dropna(how = 'all',inplace = True)

df.drop(['Günlük Test'],axis=1,inplace = True)

df.shape
plot_data = df.groupby(['vakalar'], as_index=False).ölüm.sum()



fig = px.line(plot_data, x='vakalar', y='ölüm')

fig.show()
plot_data = df.groupby(['vakalar'], as_index=False).vaka.sum()



fig = px.line(plot_data, x='vakalar', y='vaka')

fig.show()
cnt_srs = df['vakalar'].value_counts().head()

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

    title='COVID-19 by Toplam Vakalar',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="vakalar")
cnt_srs = df['ölüm'].value_counts().head()

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

    title='COVID-19 by Günlük Ölüm ',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="ölüm")
cnt_srs = df['vaka'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='COVID-19 by Günlük Vaka ',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="vaka")
# Grouping it by age and province

plot_data = df.groupby(['ölüm', 'vaka'], as_index=False).vakalar.sum()



fig = px.bar(plot_data, x='ölüm', y='vakalar', color='vaka')

fig.show()
trace1 = go.Box(

    y=df["ölüm"],

    name = 'Günlük Ölüm',

    marker = dict(color = 'rgb(0,145,119)')

)



trace2 = go.Box(

    y=df["vakalar"],

    name = 'Toplam Vakalar',

    marker = dict(color = 'rgb(5, 79, 174)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='Ölüm and Toplam Vakalar', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))



fig = dict(data=data, layout=layout)

iplot(fig)
plt.style.use('dark_background')

df.plot.area(y=['ölüm','vaka','vakalar',],alpha=0.4,figsize=(12, 6));
sns.jointplot(df['ölüm'],df['vakalar'],data=df,kind='kde',space=0,color='g')
dfcorr=df.corr()

dfcorr
plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=False,cmap='nipy_spectral')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRYpInbFyB7Rh7B4AfvkYtiJo22yEC-FH-MhKuV4VEPwpuNihou&usqp=CAU',width=400,height=400)