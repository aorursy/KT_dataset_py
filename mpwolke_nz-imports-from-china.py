#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcShGrBsiUUMal7Gx8moyU12wBMrDgD3yj-aS3KCtRZv3iUNjo2v',width=400,height=400)
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
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsimportschinacsv/imports-china.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'imports-china.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head().style.background_gradient(cmap='viridis')
df.dtypes
sns.countplot(df["Year"])

plt.xticks(rotation=90)

plt.show()
df["Year"].plot.hist()

plt.show()
df.corr()
plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='Reds')

plt.show()
cnt_srs = df['Year'].value_counts().head()

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

    title='Year distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Year")
fig = px.pie( values=df.groupby(['Year']).size().values,names=df.groupby(['Year']).size().index)

fig.update_layout(

    title = "Years",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
fig = px.histogram(df[df.Year.notna()],x="Year",marginal="box",nbins=10)

fig.update_layout(

    title = "Years",

    xaxis_title="Year",

    yaxis_title="Number of Years",

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
df_aux = df[df.Year.notna()]

df_aux=df_aux[df_aux.Year.notna()]

#df_patients_aux=df_patients_aux.Description.notna()

fig = px.histogram(df_aux,x="Year",color="Year",marginal="box",opacity=1,nbins=10)

fig.update_layout(

    title = "Year",

    xaxis_title="Year",

    yaxis_title="Description of Years",

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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'http://www.nzcta.co.nz/images/logo-cn.png',width=400,height=400)