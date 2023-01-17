# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.offline as py

from plotly.offline import init_notebook_mode,iplot

import plotly.graph_objects as go

py.init_notebook_mode(connected=True)
water=pd.read_csv('../input/india-water-quality-data/IndiaAffectedWaterQualityAreas.csv',encoding='latin1')

water.head()
water.info()
water['Quality Parameter'].unique()
col="Quality Parameter"

grouped=water[col].value_counts().reset_index()

grouped=grouped.rename(columns={"index":col,col:"count"})

grouped
trace=go.Pie(labels=grouped[col],values=grouped['count'],pull=[0.05,0])

layout=go.Layout(title="Which quality of water is more",height=600,legend=dict(x=0.1,y=1))

fig=go.Figure(data=[trace],layout=layout)

fig.show()
water['Year']=pd.to_datetime(water['Year'])

water['year_d'] = water['Year'].dt.year

water.head()
d1=water[water['Quality Parameter']=='Salinity']

d2=water[water['Quality Parameter']=='Fluoride']

d3=water[water['Quality Parameter']=='Iron']

d4=water[water['Quality Parameter']=='Arsenic']

d5=water[water['Quality Parameter']=='Nitrate']



col="year_d"



vc1=d1[col].value_counts().reset_index()

vc1=vc1.rename(columns={"index":col,col:"count"})

vc1=vc1.sort_values(col)



vc2=d2[col].value_counts().reset_index()

vc2=vc2.rename(columns={"index":col,col:"count"})

vc2=vc2.sort_values(col)



vc3=d3[col].value_counts().reset_index()

vc3=vc3.rename(columns={"index":col,col:"count"})

vc3=vc3.sort_values(col)



vc4=d4[col].value_counts().reset_index()

vc4=vc4.rename(columns={"index":col,col:"count"})

vc4=vc4.sort_values(col)



vc5=d5[col].value_counts().reset_index()

vc5=vc5.rename(columns={"index":col,col:"count"})

vc5=vc5.sort_values(col)
trace1=go.Scatter(x=vc1[col],y=vc1['count'],name="Salinity")

trace2=go.Scatter(x=vc2[col],y=vc2['count'],name="Fluoride")

trace3=go.Scatter(x=vc3[col],y=vc3['count'],name="Iron")

trace4=go.Scatter(x=vc4[col],y=vc4['count'],name="Arsenic")

trace5=go.Scatter(x=vc5[col],y=vc5['count'],name="Nitrate")

data1=[trace1,trace2,trace3,trace4,trace5]

layout=go.Layout(title="Content added over the years",legend=dict(x=0.1,y=1.1,orientation='h'),template="plotly_dark")

fig=go.Figure(data1,layout=layout)

fig.show()