import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
copy = pd.read_csv('../input/videogamesales/vgsales.csv')

data = copy.copy()

data.head()
data.info()
data.isnull().any()
import missingno as msno

msno.matrix(data);
data.dropna(inplace=True)
data.isnull().any()
import plotly.express as px

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Box(y=data['NA_Sales'],name='Sales in North America (in millions)',boxpoints='suspectedoutliers',jitter=0.3,marker_color='rgb(95, 158, 160)',line_color='rgb(95, 158, 160)'))

fig.add_trace(go.Box(y=data['EU_Sales'],name='Sales in Europe (in millions)',boxpoints='suspectedoutliers',jitter=0.3,marker_color='rgb(70, 130, 180)',line_color='rgb(70, 130, 180)'))

fig.add_trace(go.Box(y=data['JP_Sales'],name='Sales in Japan (in millions)',boxpoints='suspectedoutliers',jitter=0.3,marker_color='rgb(176, 196, 222)',line_color='rgb(176, 196, 222)'))

fig.add_trace(go.Box(y=data['Other_Sales'],name='Sales in the rest of the world (in millions)',boxpoints='suspectedoutliers',jitter=0.3,marker_color='rgb(173, 216, 230)',line_color='rgb(173, 216, 230)'))

fig.add_trace(go.Box(y=data['Global_Sales'],name='Total worldwide sales.',boxpoints='suspectedoutliers',jitter=0.3,marker_color='rgb(135, 206, 250)',line_color='rgb(135, 206, 250)'))
fig = px.bar(data['Platform'].value_counts(),color_discrete_sequence=px.colors.sequential.Bluered_r,labels={'index':'Platform',

                                                                                                           'value':'Values'})

fig.show()
fig = px.pie(data, values='Platform', names='Genre',color_discrete_sequence=px.colors.sequential.Brwnyl_r,title='Genre Distributions According to Platforms')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
sorted_data = data.sort_values(by=['Year'], ascending=True)

fig = px.line(sorted_data, x="Year", y="Global_Sales", title='Worldwide Sales by Year',labels={'Global_Sales':'Global Sales'},color_discrete_sequence=px.colors.sequential.Blackbody)

fig.show()
fig = px.histogram(data,x='Genre',color='Platform',barmode='group',title='Game Types Sold by Platform')

fig.show()
labels = ['Electronic Arts','Activision','Namco Bandai Games','Ubisoft','Konami Digital Entertainment','THQ','Nintendo','Sony Computer Entertainment','Sega','Take-Two Interactive']

values = [1339,966,928,918,823,712,696,682,632,412]

colors = ['Gainsboro ', 'LightGray ', 'Silver ','DarkGray ','DimGray  ','Gray ','LightSlateGray','SlateGray','DarkSlateGray','Black']

fig = go.Figure(data=[go.Pie(labels=labels,values=values)])

fig.update_traces(textposition='inside', textinfo='percent+label',textfont_size=15,marker=dict(colors=colors, line=dict(color='#031312', width=2)))

fig.show()