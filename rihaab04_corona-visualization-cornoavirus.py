# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import plotly.express as px



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")

df.head()
data=df.drop('Unnamed: 0' , axis=1)

data.shape
data.isna().sum() 
import plotly.graph_objects as go

rb2=data.groupby('Country')['Confirmed'].sum().reset_index()



fig = go.Figure(data=go.Choropleth(

    locations=rb2['Country'], # Spatial coordinates

    z = rb2['Confirmed'].astype(float), 

    locationmode = 'country names',

    colorscale = 'reds',

))



fig.update_layout(

    title_text = 'Number of confirmed affected people per country',

)



fig.show()
v= pd.pivot_table(data,index=["Province/State"] ,aggfunc=np.sum).reset_index()

v['sum']=v['Confirmed'] + v['Suspected']



v1=v.sort_values(by='sum', ascending=False).head(5)

mals=v1['Province/State']

fig = go.Figure(data=[

    go.Bar(name='Confirmed', x=mals, y=v1['Confirmed']),

    go.Bar(name='Deaths', x=mals, y=v1['Deaths']),

    go.Bar(name='Recovered', x=mals, y=v1['Recovered']) ,

    go.Bar(name='Suspected', x=mals, y=v1['Suspected']) 



])



# Change the bar mode

fig.update_layout(barmode='stack', title ='Top 5 highest touched zones ')

fig.show()
p=v.sort_values(by='sum', ascending=True).iloc[1:4,:] #i excluded Tipot because it got 0 everywhere .. smh

px.scatter(p, x='Deaths', y='Confirmed' , color='Recovered',facet_col='Province/State' , title=' 3 lowest touched zones ' )
rb=data.groupby('Date last updated')['Confirmed'].sum().reset_index()

rb1=data.groupby('Date last updated')['Deaths'].sum().reset_index()

nn=pd.merge(rb1,rb, on='Date last updated', how='inner')

px.scatter(nn , x='Date last updated', y='Confirmed', size='Confirmed' , color='Deaths' , title='Dates review'  )

import plotly.graph_objects as go

rb0=data.groupby('Date last updated')['Recovered'].sum().reset_index()

px.line(rb0 , x='Date last updated', y='Recovered', title='Recovered people per period')
r0=data.groupby('Country')['Recovered'].sum().reset_index().max()

print(r0[0],'is the country with most recovered cases:  ',r0[1]  ) 
import plotly.graph_objects as go

a=data.loc[df.Confirmed == max(data.Confirmed)]

mals=a['Date last updated']

fig = go.Figure(data=[

    go.Bar(name='Confirmed', x=mals, y=a['Confirmed']),

    go.Bar(name='Deaths', x=mals, y=a['Deaths']) 

])

# Change the bar mode

fig.update_layout(barmode='stack', title ='highest Confirmed cases ',)

fig.show()
