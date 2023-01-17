import plotly.graph_objects as go
import numpy as np
N = 100
t = np.linspace(0, 10, N)
y = np.sin(t)
fig = go.Figure(data=go.Scatter(x=t,y=y,mode='markers'))
fig.show()
#with seed random value wont change
np.random.seed(1)
N =100
#Set values for X and Y axis, here as we will show 3 charts so need to create data for 3 Y-axis
random_x = np.linspace(0,1,N)
random_y = np.random.rand(N)+5
random_y0 = np.random.rand(N)
random_y1=np.random.rand(N)-5
fig = go.Figure()
#Scatter chart
fig.add_trace(go.Scatter(x=random_x,y=random_y,mode='markers',name='markers'))
#Line plus marker chart
fig.add_trace(go.Scatter(x=random_x, y=random_y0,mode='lines+markers',name='lines+markers'))
#line chart
fig.add_trace(go.Scatter(x=random_x,y=random_y1,mode='lines',name='lines'))
fig.show()
fig=go.Figure(data=go.Scatter(
x=[1,2,3,4],
y=[10,11,12,13,14],
mode='markers',marker=dict(size=[40,60,80,100],color=[1,2,3,4])))
fig.update_layout(title='Bubble Chart')
fig.show()
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math 

#Continet level data
data = px.data.gapminder()
#Fetch the data for year 2007
df_2007=data[data['year']==2007]
#Sort w.r.t to continet and country
df_2007 = df_2007.sort_values(['continent','country'])
#print(df_2007)

hover_text=[]
bubble_size=[]

#loop to add text in hover for all df_2007 data
for index, row in df_2007.iterrows():
    hover_text.append(('Country: {country}<br>'+
                      'Life Expectancy: {lifeExp}<br>'+
                      'GDP per capita: {gdp}<br>'+
                      'Population: {pop}<br>'+
                      'Year: {year}').format(country=row['country'],
                                            lifeExp=row['lifeExp'],
                                            gdp=row['gdpPercap'],
                                            pop=row['pop'],
                                            year=row['year']))
    bubble_size.append(math.sqrt(row['pop']))

df_2007['text'] = hover_text
df_2007['size'] = bubble_size

sizeref = 2.*max(df_2007['size'])/(100**2)

# Dictionary with dataframes for each continent

continent_names = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']
#Fetch all the continent data
continent_data = {continent:df_2007.query("continent == '%s'" %continent)
                              for continent in continent_names}
fig = go.Figure()

colors = ['#a3a7e4'] * 100

#Design  graph by passing X, Y axis details plus size of the markers
for continent_name, continent in continent_data.items():
    fig.add_trace(go.Scatter(x=continent['gdpPercap'],
                             y=continent['lifeExp'],
                             name=continent_name,text=continent['text'],
                             marker_size=continent['size']))
    
    #Convert lines in to bubble chart by setting mode='markers'
    fig.update_traces(mode='markers',marker=dict(sizemode='area',
                                             sizeref=sizeref,
                                             line_width=2))
    
    
# Set the title for X and Y axis also some formatting
fig.update_layout(
    title='Life Expectancy v. Per Capita GDP, 2007',
    #create dictionary for xaxis and yaxis
    xaxis=dict(
        title="GDP per capita (2000 dollars)",
        gridcolor='black',
        gridwidth=2
    ),
    yaxis=dict(
        title="Life Expectancy (years)",
        gridcolor='black',
        gridwidth=2
    ),
    #set the same color of Hover, it wont change as per bubble color
    hoverlabel = dict(
        bgcolor = 'rgb(252,141,89)' #whatever format you want
    ),
    #change the background color
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
#Display the graph
fig.show()
        
#print(help(go.Scatter.hoverlabel))