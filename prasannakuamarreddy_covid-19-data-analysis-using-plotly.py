# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from matplotlib import style



import plotly

import plotly.express as px

import plotly.graph_objects as go

#import chart_studio.plotly as py



import cufflinks as cf

import folium



import plotly.offline as pyo

from plotly.offline import init_notebook_mode,plot,iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(r"/kaggle/input/covid19indiastatewiseape26/state_wise.csv")

df
ind_df = pd.read_excel(r"/kaggle/input/coordinates-of-india/Indian Coordinates.xlsx")

ind_df
ind_df.rename(columns={'Name of State / UT':'State'},inplace=True)

ind_df
df2 = pd.merge(ind_df,df,on='State')
#Once complete you can comment it out

#df.drop(['State_Notes'],axis = 1,inplace=True)

#df.drop([0],axis = 0, inplace = True)
df.style.background_gradient(cmap='Greens')
Active_tbl = df.groupby('State')['Active'].sum().sort_values(ascending= False).to_frame()

Active_tbl
Active_tbl.style.background_gradient(cmap='Oranges')
df.sort_values(ascending=True,by='Active',inplace=True)

df
#Pandas

df.plot(kind='bar',x = 'State', y = 'Active')

plt.show()







#PlotLy

trace1 = go.Bar(

    x = df['State'],

    y = df['Active'],

    name = 'No of cases state wise',

    )

data = [trace1]

layout = go.Layout(barmode = "group",)

fig = go.Figure(data = data, layout = layout)

iplot(fig)

#px

fig_px = px.bar(df, x='State',y='Active')

fig_px.show()
#Line graph using px



fig = px.line(x = df['State'],y = df['Active'],labels = {'x':'State','y':'Total Active case now'})

fig.show()
#px = colored



fig_px = px.bar(df, x='State',y='Active',

               hover_data = ['Confirmed','Deaths'],

               color = 'Deaths')

fig_px.show()
#scatter plot



fig = px.scatter(df,x="State",y="Active")

fig.show()
#Stack bar plot

fig = go.Figure(

        data=[

            go.Bar(name ='Active cases',x=df['State'], y=df['Active'], offsetgroup=0),

            go.Bar(name ='Total case', x = df['State'], y=df['Confirmed'], offsetgroup=1)

        ])

fig.update_layout(barmode = 'stack')

fig.show()



#go with line and scatter plots



fig = go.Figure()



fig.add_trace(go.Scatter(x=df['State'],y=df['Deaths'],mode='lines',name='Deaths'))

fig.add_trace(go.Scatter(x=df['State'],y=df['Active'],mode='lines+markers',name='Active cases'))

fig.add_trace(go.Scatter(x=df['State'],y=df['Confirmed'],mode='markers',name='Confirmed cases'))

fig.add_trace(go.Scatter(x=df['State'],y=df['Recovered'],mode='lines',name='Confirmed cases'))

fig.show()
#Bubble scater plot



fig = go.Figure(data = 

                go.Scatter(x=df['State'],y=df['Active'],mode = 'markers',

                marker = dict(size = df['Active']/50,

                             color = df['Recovered'])))

fig.show()
map=folium.Map(location=[20,70],zoom_start=4,tiles='Stamenterrain')



for lat,long,value, name in zip(df2['Latitude'],df2['Longitude'],df2['Active'],df2['State']):

    folium.CircleMarker([lat,long],radius=value*0.01,popup=('<strong>State</strong>: '+str(name).capitalize()+'<br>''<strong>Total Cases</strong>: ' + str(value)+ '<br>'),color='red',fill_color='red',fill_opacity=0.003).add_to(map)

    

map