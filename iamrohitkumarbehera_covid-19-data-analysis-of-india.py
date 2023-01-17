import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pylab as plt
%matplotlib inline
import statsmodels.api as sm
import os
from pandas import datetime

india_covid = pd.read_csv("../input/statewise_data_with_new_cases.csv") 
india_covid
sorted_country_df = india_covid.sort_values('Confirmed', ascending= False)
sorted_country_df_d = india_covid.sort_values('Deaths', ascending= False)
sorted_country_df_r = india_covid.sort_values('Recovered', ascending= False)
def highlight_col(x):
    r = 'background-color: red'
    m = 'background-color: magenta'
    g = 'background-color: grey'
    y = 'background-color: yellow'
    o = 'background-color: orange'
    l = 'background-color: lime'
    gy = 'background-color: greenyellow'
    b = 'background-color: skyblue'
    b1= 'background-color: bisque'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1.iloc[:, 3] = b
    df1.iloc[:, 4] = b1
    df1.iloc[:, 5] = r
    df1.iloc[:, 6] = gy
    return df1
from ipywidgets import interact, interactive, fixed, interact_manual
def show_latest_cases(n):
    n = int(n)
    return india_covid.sort_values('Confirmed', ascending= False).head(n).style.apply( highlight_col,axis=None)

interact(show_latest_cases, n='10')
import plotly.express as px
px.bar(
    sorted_country_df.head(15),
    x = "State/UT",
    y = "Confirmed",
    title= "Top 10 worst affected state confirmed case in india", # the axis names
    color_discrete_sequence=["blue"], 
    height=500,
    width=800
)
import plotly.express as px
px.bar(
    sorted_country_df_d.head(15),
    x = "State/UT",
    y = "Deaths",
    title= "Top 10 worst affected state death case in india", # the axis names
    color_discrete_sequence=["red"], 
    height=500,
    width=800
)
from ipywidgets import interact, interactive, fixed
def bubble_chart(n):
    fig = px.scatter(sorted_country_df.head(n), x="State/UT", y="Confirmed", size="Confirmed", color="State/UT",
               hover_name="State/UT", size_max=30)
    fig.update_layout(
    title=str(n) +" Worst hit confirmcases state in india",
    xaxis_title="States",
    yaxis_title="Confirmed Cases",
    width =700
    )
    fig.show()

interact(bubble_chart, n=10)
from ipywidgets import interact, interactive, fixed
def bubble_chart(n):
    fig = px.scatter(sorted_country_df.head(n), x="State/UT", y="Deaths", size="Deaths", color="State/UT",
               hover_name="State/UT", size_max=30)
    fig.update_layout(
    title=str(n) +" Worst hit death in india",
    xaxis_title="States",
    yaxis_title="Deaths",
    width =700
    )
    fig.show()

interact(bubble_chart, n=10)
from ipywidgets import interact, interactive, fixed
def bubble_chart(n):
    fig = px.scatter(sorted_country_df.head(n), x="State/UT", y="Recovered", size="Recovered", color="State/UT",
               hover_name="State/UT", size_max=30)
    fig.update_layout(
    title=str(n) +" Top State Recovered in india",
    xaxis_title="States",
    yaxis_title="Recovered",
    width =700
    )
    fig.show()

interact(bubble_chart, n=10)
import folium
import numpy as np
world_map = folium.Map(location=[21,78], tiles="cartodbpositron", zoom_start=4, max_zoom = 20, min_zoom = 2)
# iterate over all the rows of confirmed_df to get the lat/long
for i in range(0,len(india_covid)):
    folium.Circle(
        location=[india_covid.iloc[i]['Latitude'], india_covid.iloc[i]['Longitude']],
        fill=True,
        radius=(int((np.log(india_covid.iloc[i,-1]+1.00001)))+0.2)*50000,
        color='red',
        fill_color='indigo',
    ).add_to(world_map)
    
world_map