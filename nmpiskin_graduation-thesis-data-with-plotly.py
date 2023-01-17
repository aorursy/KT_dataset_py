# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
timesData = pd.read_excel("../input/bitirme-100-bin.xlsx")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
timesData=timesData.set_index("id")
timesData.head()
timesData=timesData.drop(["basinc","yagmur"],axis=1)

timesData.head()
timesData.zaman=pd.to_datetime(timesData.zaman)
df=timesData.set_index("zaman","id")
df=df.resample("60S").mean()
dataframe=df.reset_index()
dataframe.corr()
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()
data=dataframe.copy()
#creatin trace1
trace1=go.Scatter(
    x=data.ruzgarhizi,
    y=data.xivme,
    mode="lines",
    name="xivme",
    marker=dict(color='rgba(16,112,2,0.8)'))
#Creating trace2
trace2=go.Scatter(
    x=data.ruzgarhizi,
    y=data.yivme,
    mode="lines+markers",
    name="yivme",
    marker=dict(color='rgba(80,26,80,0.8)'))
data=[trace1,trace2]
layout=dict(title='Time graph of the acceleration of the tower in x and y direction',
           xaxis=dict(title="Rüzgar Hızı",ticklen=5,zeroline=False))
fig=dict(data=data,layout=layout)
iplot(fig)
data=dataframe.copy()
trace1=go.Scatter(
    x=data.ruzgarhizi,
    y=data.xivme,
    mode="markers",
    name="xivme",
    marker=dict(color='rgba(16,112,2,0.8)'))
#Creating trace2
trace2=go.Scatter(
    x=data.ruzgarhizi,
    y=data.yivme,
    mode="markers",
    name="yivme",
    marker=dict(color='rgba(80,26,80,0.8)'))
data=[trace1,trace2]
layout=dict(title='Time graph of the acceleration of the tower in x and y direction',
           xaxis=dict(title="Rüzgar Hızı",ticklen=5,zeroline=False))
fig=dict(data=data,layout=layout)
iplot(fig)
timesData.describe()
data1=dataframe[dataframe.ruzgarhizi > 20].iloc[:3,:]
data2=dataframe[dataframe.ruzgarhizi < 5].iloc[:3,:]
data1
data1=dataframe[dataframe.ruzgarhizi > 20].iloc[:3,:]
data2=dataframe[dataframe.ruzgarhizi < 10].iloc[:3,:]
sample=["1","2","3"]
trace1=go.Bar(
    x=sample,
    y=data1.xivme,
    name="xivmeyüksek",
    marker=dict(color='rgba(255,174,255,0.5)',
               line=dict(color='rgba(0,0,0)',width=1.5)),)
trace2=go.Bar(
    x=sample,
    y=data2.xivme,
    name="xivmedüşük",
    marker=dict(color= 'rgba(255, 255, 128, 0.5)',
               line=dict(color='rgba(0,0,0)',width=1.5)),)
data=[trace1,trace2]
layout=go.Layout(barmode="group")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
data=dataframe.copy()
trace1=go.Histogram(
    x=data.ruzgaryonu,
    opacity=0.75,
    name = "2011",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
data=[trace1]
ayout = go.Layout(barmode='overlay',
                   title=' students-staff ratio in 2011 and 2012',
                   xaxis=dict(title='students-staff ratio'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
data=dataframe.copy()
trace0=go.Box(
    y=data.sg1,
    name='Number 1 Strain Gage',
    marker=dict(
        color='rgb(12,12,148)',
    )
)
trace1=go.Box(
    y=data.sg2,
    name='Number 2 Strain Gage',
    marker=dict(
        color='rgb(12,128,128)',))
data=[trace0,trace1]
iplot(data)
import plotly.figure_factory as ff
data=dataframe.copy()
data=dataframe.loc[:,["nem","sicaklik"]]
data["index"]=np.arange(1,len(data)+1)

fig=ff.create_scatterplotmatrix(data,diag='box',index='index',colormap='Portland',
                               colormap_type='cat',
                               height=700,width=700)
iplot(fig)
data=dataframe[dataframe.ruzgarhizi > 15]
trace1=go.Scatter3d(
    x=data.xivme,
    y=data.yivme,
    z=data.zivme,
    mode='markers',
    marker=dict(
        size=10,
        color=data.ruzgarhizi,
        colorscale='Viridis',
        opacity=0.8
    )
)
data=[trace1]
layout=go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig=go.Figure(data=data,layout=layout)
iplot(fig)
data=dataframe.copy()
trace1=go.Scatter(
    x=data.ruzgarhizi,
    y=data.xivme,
    name='Xivme',
    mode="markers"
)
trace2=go.Scatter(
    x=data.ruzgarhizi,
    y=data.yivme,
    xaxis='x2',
    yaxis='y2',
    name="Yivme",
    mode="markers"
)
trace3=go.Scatter(
    x=data.sicaklik,
    y=data.nem,
    xaxis='x3',
    yaxis='y3',
    name='Humidity vs temperature'
    
)

data=[trace1,trace2,trace3]
layout=go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),

    title="Humidity vs temperature and xivme and y ivme vs rüzgarhizi"
)
fig= go.Figure(data=data,layout=layout)
iplot(fig)