# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd
import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go


import warnings
warnings.filterwarnings("ignore")
co2=pd.read_csv("/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv")
df=co2.copy()
df.head(2)
df.info()
df.shape
df.isnull().values.any()
df.isnull().sum()
df["Entity"].unique()
df[df.duplicated() == True]
df=df.rename(columns={"Annual CO₂ emissions (tonnes )":"CO2"})
df.head(2)
df.drop(['Code'],inplace=True,axis=1)
df.head(2)
df_World=df[df["Entity"]=="World"]
fig = go.Figure(data=go.Scatter(x=df_World['Year'],
                                y=df_World['CO2'],
                                mode='lines')) # hover text goes here
fig.update_layout(title='Number Of CO2  Emission  In World',title_x=0.5,xaxis_title="Year",yaxis_title="Number of CO2 Emission (tonnes)")
fig.show()
df_China=df[df["Entity"]=="China"]
fig = go.Figure(data=go.Scatter(x=df_China['Year'],
                                y=df_China['CO2'],
                                mode='lines',
                               marker_color='darkred')) 

fig.update_layout(title='Number of CO2 Emission In China Over Time',xaxis_title="Year",yaxis_title="Number of CO2 Emission",xaxis_range=['1990','2017'])
fig.show()
fig = go.Figure()

df_Turkey=df[df["Entity"]=="Turkey"]
df_Germany=df[df["Entity"]=="Germany"]
df_France=df[df["Entity"]=="France"]

fig.add_trace(go.Scatter(x=df_Turkey['Year'], y=df_Turkey['CO2'], name = 'Turkey-Dot',
                         line=dict(color='royalblue', width=4,dash="dot")))

fig.add_trace(go.Scatter(x=df_Germany['Year'], y=df_Germany['CO2'], name = 'Germany-Dashdot',
                         line=dict(color='green', width=4,dash="dashdot")))

fig.add_trace(go.Scatter(x=df_France['Year'], y=df_France['CO2'], name = 'France-Dash',
                         line=dict(color='brown', width=4,dash="dash")))
fig.update_layout(title='Co2 Emission Over Time For Fifferent Countries',title_x=0.5,xaxis_title="Years",yaxis_title="Number of Co2 Emission(tonnes)")
fig.show()
temp = df.groupby('Entity').sum().reset_index()
fig = px.treemap(temp,path = ['Entity'],values = 'CO2')
fig.update_layout(title='Co2 Emission ',title_x=0.5)
fig.show()

df_EU_28=df[df["Entity"]=="EU-28"]
df_United_States=df[df["Entity"]=="United States"]
df_China=df[df["Entity"]=="China"]
df_Aspa=df[df["Entity"]=="Asia and Pacific (other)"]
df_EU_O=df[df["Entity"]=="Europe (other)"]
df_Russia=df[df["Entity"]=="Russia"]

Russia_total=df_Russia["CO2"][df_Russia["Year"]>=2010].sum()
EU_28_total=df_EU_28["CO2"][df_EU_28["Year"]>=2010].sum()
United_States_total=df_United_States["CO2"][df_United_States["Year"]>=2010].sum()
China_total=df_China["CO2"][df_China["Year"]>=2010].sum()
Aspa_total=df_Aspa["CO2"][df_Aspa["Year"]>=2010].sum()
EU_O_total=df_EU_O["CO2"][df_EU_O["Year"]>=2010].sum()

total_data={"Eu-28":EU_28_total,
           "Russia":Russia_total,
           "United States":United_States_total,
           "China":China_total,
           "Asia and Pacific (other)":Aspa_total,
           "Europe (other)":EU_O_total}

df_total = pd.DataFrame(data=total_data,index=["Total"])
df_total=df_total.transpose()

fig = px.bar(df_total, x=["Eu-28", "Russia", "United States", "China ", "Asia and Pacific (other)", "Europe (other)"],
             y=df_total["Total"])
fig.update_layout(title="Emission Amount vs Entity",
                  title_x=0.5,
                  xaxis_title="Entity",
                  yaxis_title="Total Co2 Emission 2010-2017")

fig.show()
df_2010=df[df["Year"]==2010]
df_2010=df_2010.drop([20612])# World
df_2010=df_2010.sort_values("CO2",ascending=False)[:10]

fig = go.Figure(go.Bar(
    x=df_2010['Entity'],y=df_2010['CO2'],
    marker={'color': df_2010['CO2'], 
    'colorscale': 'Viridis'},  
    text=df_2010['CO2'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10  Entity CO2 Emission 2010',
                  title_x=0.5,
                  yaxis_title="CO2 Emission Count (tonnes )",
                  xaxis_title=" Entity")
fig.show()
df_2010=df[df["Year"]==2010]
df_2010=df_2010.drop([20612])
df_2010=df_2010.sort_values("CO2",ascending=False)[:10]
fig = go.Figure([go.Pie(labels=df_2010['Entity'], values=df_2010['CO2'],
                        pull=[0.2, 0.1, 0, 0],
                        hole=0.3)])  # can change the size of hole 

fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=15)
fig.update_layout(title="2010 CO2 Emission",title_x=0.5,
                 annotations=[dict(text='CO2', x=0.50, y=0.5, font_size=20, showarrow=False)])
fig.show()
df_2010=df[df["Year"]==2010]
word_info=df_2010[df_2010["Entity"]=="World"]
word_co2=word_info["CO2"].values

df_2010["ratio"]=(df_2010["CO2"]/word_co2)*100
df_2010=df_2010.drop([20612])# world 
df_2010=df_2010.sort_values("ratio",ascending=False)[:10]

                     
fig = go.Figure(go.Funnel(
    y=df_2010["Entity"],
    x=df_2010["ratio"] ))
fig.update_layout(title='2010 CO2 Emission Ratio ',xaxis_title="Ratio",yaxis_title=" Entity ",title_x=0.5)
fig.show()
fig = go.Figure(data=[go.Scatter(
    x=df_2010['Entity'], y=df_2010['ratio'],
    mode='markers',
    marker=dict(
        color=df_2010['ratio'],
        size=df_2010['ratio']*3,
        showscale=True
    ))])

fig.update_layout(title='2010 CO2 Emission Ratio ',xaxis_title="Entity",yaxis_title=" Ratio ",title_x=0.5)
fig.show()
# Multiple Bullet

df_2010=df[df["Year"]==2010]
df_2010=df_2010.drop([20612])# word
df_2010=df_2010.drop([17637])#(-) value ? 
df_2010=df_2010.sort_values("CO2",ascending=False)

CO2_avg=df_2010.CO2.mean()

CO2_min=df_2010.CO2.min()

CO2_max=df_2010.CO2.max()

CO2_sum=df_2010.CO2.sum()



fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = CO2_avg,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Mean CO2 Emission 2010",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,250000000]},
        'bar': {'color': "cyan"}}
))


fig.add_trace(go.Indicator(
    mode = "number+gauge", value = CO2_min,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Min CO2 Emission 2010",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,5000]},
        'bar': {'color': "cyan"}}
))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = CO2_max,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text' :"Max CO2 Emission 2010",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,10000000000]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = CO2_sum,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Total CO2 Emission 2010",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,55000000000]},
        'bar': {'color': "darkcyan"}}
))

fig.update_layout(title=" 2010 CO2 Emission Statistics ",title_x=0.5)
fig.show()