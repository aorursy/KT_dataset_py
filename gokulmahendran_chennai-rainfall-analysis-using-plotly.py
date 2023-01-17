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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as pyo

import plotly.graph_objs as go

import plotly.figure_factory as ff

import warnings

warnings.filterwarnings(action="ignore")

import plotly

from plotly import tools

pyo.init_notebook_mode(connected=True)
res_df=pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv")



res_df["Date"]=pd.to_datetime(res_df["Date"],format="%d-%m-%Y")



data=[(go.Scatter(x=res_df["Date"],y=res_df[i],mode="lines",name=i)) for i in res_df.columns[1:]]



layout=go.Layout(title=dict(text="Reserviors Level of Chennai"),width=1000)



fig=go.Figure(data=data,layout=layout)

pyo.iplot(fig)
titles=["Water levels in "+i+" Reservoir" for i in res_df.columns[1:]]

fig=plotly.tools.make_subplots(rows=4,cols=1,subplot_titles=titles,vertical_spacing=.03)

r=1

c=1

for i in data:

    fig.append_trace(i,row=r,col=c)

    r+=1

fig["layout"].update(height=1800,width=1000)



pyo.iplot(fig)
res_df["year"]=res_df["Date"].dt.year

res_df["month"]=res_df["Date"].dt.month

mean_df=res_df.groupby(["year","month"]).mean().reset_index()



fig=plotly.tools.make_subplots(rows=1,cols=3,subplot_titles=['POONDI', 'REDHILLS', 'CHEMBARAMBAKKAM'])

c=1

for i in ['POONDI', 'REDHILLS', 'CHEMBARAMBAKKAM']:

    df=pd.pivot(data=mean_df,index="month",columns="year",values=i)

    df.index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig.append_trace(go.Heatmap(y=df.index,

                                x=df.columns,

                                z=df,

                                zmax=res_df.iloc[:,1:5].values.max(),

                                zmin=res_df.iloc[:,1:5].values.min(),

                            ),1,c)

    c+=1

fig["layout"].update(width=1000)

pyo.iplot(fig)
df=pd.pivot(data=mean_df,index="month",columns="year",values="CHOLAVARAM")

df.index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
data=[go.Heatmap(z=df,y=df.index,x=df.columns,name="CHOLAVARAM")]

layout=go.Layout(xaxis=dict(title="Year",tickangle=45),yaxis=dict(title="Month"

                            ))

fig=go.Figure(data=data,layout=layout)

pyo.iplot(fig)
rain_df=pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_rainfall.csv")



rain_df["Date"]=pd.to_datetime(rain_df["Date"],format="%d-%m-%Y")



data=[(go.Scatter(x=rain_df["Date"],y=rain_df[i],mode="lines",name=i)) for i in rain_df.columns[1:]]



layout=go.Layout(title=dict(text="Rainfall Level in Reservoirs"),width=1000)



fig=go.Figure(data=data,layout=layout)

pyo.iplot(fig)
rain_df["year"]=rain_df["Date"].dt.year

rain_df["month"]=rain_df["Date"].dt.month
rainfall_2015=rain_df[(rain_df["year"]==2015)&((rain_df["month"]==11)|(rain_df["month"]==12))]



data=[(go.Scatter(x=rainfall_2015["Date"],y=rainfall_2015[i],mode="lines",name=i)) for i in rainfall_2015.columns[1:5]]



layout=go.Layout(title=dict(text="Rainfall Level in Reservoirs"),width=1000)



fig=go.Figure(data=data,layout=layout)

pyo.iplot(fig)
rainfall_2015=res_df[(res_df["year"]==2015)&((res_df["month"]==11)|(res_df["month"]==12))]



data=[(go.Scatter(x=rainfall_2015["Date"],y=rainfall_2015[i],mode="lines",name=i)) for i in rainfall_2015.columns[1:5]]



layout=go.Layout(title=dict(text="Reservoirs Level"),width=1000)



fig=go.Figure(data=data,layout=layout)

pyo.iplot(fig)
mean_df=rain_df.groupby(["year","month"]).mean().reset_index()



fig=plotly.tools.make_subplots(rows=1,cols=4,subplot_titles=['POONDI', 'REDHILLS', 'CHEMBARAMBAKKAM',"CHOLAVARAM"])

c=1

for i in ['POONDI', 'REDHILLS', 'CHEMBARAMBAKKAM',"CHOLAVARAM"]:

    df=pd.pivot(data=mean_df,index="month",columns="year",values=i)

    df.index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig.append_trace(go.Heatmap(y=df.index,

                                x=df.columns,

                                z=df,

                                zmin=0,

                                zmid=30,

                                zmax=40

                            ),1,c)

    c+=1

fig["layout"].update(width=1000)

pyo.iplot(fig)