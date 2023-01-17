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


!pip install plotly

!pip install folium



!pip install ipywidgets

!pip install ipympl
from __future__ import print_function

from ipywidgets import interact, interactive, fixed, interact_manual

from IPython.core.display import display, HTML



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import folium

import plotly.graph_objects as go

import seaborn as sns

import ipywidgets as widgets

%matplotlib widget

from plotly.offline import init_notebook_mode, plot





init_notebook_mode()
death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

country_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
country_df.columns = map(str.lower, country_df.columns)

confirmed_df.columns = map(str.lower, confirmed_df.columns)

death_df.columns = map(str.lower, death_df.columns)

recovered_df.columns = map(str.lower, recovered_df.columns)



# changing province/state to state and country/region to country

confirmed_df = confirmed_df.rename(columns={'province/state': 'state', 'country/region': 'country'})

recovered_df = recovered_df.rename(columns={'province/state': 'state', 'country/region': 'country'})

death_df = death_df.rename(columns={'province/state': 'state', 'country/region': 'country'})

country_df = country_df.rename(columns={'country_region': 'country'})
print(confirmed_df.shape,death_df.shape,recovered_df.shape,country_df.shape)

confirmed_df.head(2)
confirmed_total = int(country_df['confirmed'].sum())

deaths_total = int(country_df['deaths'].sum())

recovered_total = int(country_df['recovered'].sum())

active_total = int(country_df['active'].sum())
display(HTML("<div style = 'background-color: #504e4e; padding: 30px '>" +

             "<span style='color: #fff; font-size:30px;'> Confirmed: "  + str(confirmed_total) +"</span>" +

             "<span style='color: red; font-size:30px;margin-left:20px;'> Deaths: " + str(deaths_total) + "</span>"+

             "<span style='color: lightgreen; font-size:30px; margin-left:20px;'> Recovered: " + str(recovered_total) + "</span>"+

             "</div>")

       )
fig = go.FigureWidget( layout=go.Layout() )

def highlight_col(x):

    r = 'background-color: red'

    y = 'background-color: purple'

    g = 'background-color: grey'

    df1 = pd.DataFrame('', index=x.index, columns=x.columns)

    df1.iloc[:, 4] = y

    df1.iloc[:, 5] = r

    df1.iloc[:, 6] = g

    

    return df1



def show_latest_cases(n):

    n = int(n)

    return country_df.sort_values('confirmed', ascending= False).head(n).style.apply(highlight_col, axis=None)

print("Change Value of N")

interact(show_latest_cases, n='10')



# ipywLayout = widgets.Layout(border='solid 2px green')

# ipywLayout.display='none' # uncomment this, run cell again - then the graph/figure disappears

# widgets.VBox([fig], layout=ipywLayout)

sorted_country_df = country_df.sort_values('confirmed', ascending= False)
def bubble_chart(n):

    if n>=1:

        fig = px.scatter(sorted_country_df.head(n), x="country", y="confirmed", size="confirmed", color="country",

                   hover_name="country", size_max=60)

        fig.update_layout(

        title=str(n) +" Worst hit countries",

        xaxis_title="Countries",

        yaxis_title="Confirmed Cases",

        width = 850

        )

        fig.show();

    else:

        print("invalid Input")



interact(bubble_chart, n=10)



# ipywLayout = widgets.Layout(border='solid 2px green')

# # ipywLayout.display='none'

# widgets.VBox([fig], layout=ipywLayout)
sorted_country_on_death_df = country_df.sort_values('deaths', ascending= False)

def bubble_chart_on_death(n):

    if n>=1:

        fig = px.scatter(sorted_country_on_death_df.head(n), x="country", y="deaths", size="deaths", color="country",

                   hover_name="country", size_max=60)

        fig.update_layout(

        title=str(n) +" Worst hit on death countries",

        xaxis_title="Countries",

        yaxis_title="Deaths Cases",

        width = 850

        )

        fig.show();

    else:

        print("invalid Input")



interact(bubble_chart_on_death, n=10)
sorted_country_on_recovered_df = country_df.sort_values('recovered', ascending= False)

def bubble_chart_on_recovered(n):

    if n>=1:

        fig = px.scatter(sorted_country_on_recovered_df.head(n), x="country", y="recovered", size="recovered", color="country",

                   hover_name="country", size_max=60)

        fig.update_layout(

        title=str(n) +" Highest recovry seen countries",

        xaxis_title="Countries",

        yaxis_title="recovered Cases",

        width = 850

        )

        fig.show();

    else:

        print("invalid Input")



interact(bubble_chart_on_recovered, n=10)
confirmed_df.head(2)
confirmed_df.tail(2)
def plot_cases_of_a_country(country):

    country = country.capitalize()

    labels = ['confirmed', 'deaths','recovered']

    colors = ['blue', 'red','green']

    mode_size = [6, 8,1]

    line_size = [4, 5, 1]

#     labels = ['confirmed']

#     colors = ['blue']

#     mode_size = [6]

#     line_size = [4]

    

#     df_list = [confirmed_df]

    df_list = [confirmed_df, death_df,recovered_df]

    

    

    fig = go.Figure();

    

    for i, df in enumerate(df_list):

        if country == 'World' or country == 'world':

            x_data = np.array(list(df.iloc[:, 20:].columns))

            y_data = np.sum(np.asarray(df.iloc[:,20:]),axis = 0)

            

        else:    

            x_data = np.array(list(df.iloc[:, 20:].columns))

            y_data = np.sum(np.asarray(df[df['country'] == country].iloc[:,20:]),axis = 0)

            

        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers',

        name=labels[i],

        line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,

        text = "Total " + str(labels[i]) +": "+ str(y_data[-1])

        ));

    fig.update_layout(

        title="COVID 19 cases of " + country,

        xaxis_title='Date',

        yaxis_title='No. of Cases',

        margin=dict(l=20, r=20, t=40, b=20),

        paper_bgcolor="lightgrey",

        width = 850,

        

    );

    

    fig.update_yaxes(type="linear")

    fig.show();
interact(plot_cases_of_a_country, country='India')
def plot_daily_cases_of_a_country(country):

    country = country.capitalize()

    labels = ['confirmed']

    colors = ['blue']

    mode_size = [6]

    line_size = [4]



    df_list = [confirmed_df]

    

    

    fig = go.Figure();

    

    for i, df in enumerate(df_list):

        if country == 'World' or country == 'world':

            x_data = np.array(list(df.iloc[:, 20:].columns))

            y_data = np.sum(np.asarray(df.iloc[:,20:]),axis = 0)

            

        else:    

            x_data = np.array(list(df.iloc[:, 20:].columns))

            y_data = np.sum(np.asarray(df[df['country'] == country].iloc[:,20:]),axis = 0)

        y_daily = []

        y_daily.append(y_data[0])

        for j in range(1,len(y_data)):

            y_daily.append(y_data[j] - y_data[j-1])

        fig.add_trace(go.Bar(

            x=x_data,

            y=y_daily,

            name='confirmed cases',

            marker_color='red'

        ))

    

    fig.update_layout(

        title="COVID 19 cases of " + country,

        xaxis_title='Date',

        yaxis_title='No. of Confirmed Cases',

        margin=dict(l=20, r=20, t=40, b=20),

        paper_bgcolor="lightgrey",

        width = 850,

        

    );

    

    fig.update_yaxes(type="linear")

    fig.show();
interact(plot_daily_cases_of_a_country, country='world')
def plot_daily_cases_of_a_death_country(country):

    country = country.capitalize()

    labels = ['recovered','death']

    colors = ['green','red']

    mode_size = [6,5]

    line_size = [4,4]



    df_list = [recovered_df,death_df]

    

    

    fig = go.Figure();

    

    for i, df in enumerate(df_list):

        if country == 'World' or country == 'world':

            x_data = np.array(list(df.iloc[:, 20:].columns))

            y_data = np.sum(np.asarray(df.iloc[:,20:]),axis = 0)

            

        else:    

            x_data = np.array(list(df.iloc[:, 20:].columns))

            y_data = np.sum(np.asarray(df[df['country'] == country].iloc[:,20:]),axis = 0)

        y_daily = []

        y_daily.append(y_data[0])

        for j in range(1,len(y_data)):

            y_daily.append(y_data[j] - y_data[j-1])

        fig.add_trace(go.Bar(

            x=x_data,

            y=y_daily,

            name=labels[i],

            marker_color=colors[i]

        ))

    

    fig.update_layout(

        title="COVID 19 cases of " + country,

        xaxis_title='Date',

        yaxis_title='No. of Confirmed Cases',

        margin=dict(l=20, r=20, t=40, b=20),

        paper_bgcolor="lightgrey",

        width = 850,

        

    );

    

    fig.update_yaxes(type="linear")

    fig.show();
interact(plot_daily_cases_of_a_death_country, country='world')
px.bar(

    sorted_country_df.head(10),

    x = "country",

    y = "confirmed",

    title= "Top 10 worst affected countries", # the axis names

    color_discrete_sequence=["pink"], 

    height=500,

    width=800

)
px.bar(

    sorted_country_df.head(10),

    x = "country",

    y = "deaths",

    title= "Top 10 worst affected countries", # the axis names

    color_discrete_sequence=["pink"], 

    height=500,

    width=800

)
px.bar(

    sorted_country_df.head(10),

    x = "country",

    y = "recovered",

    title= "Top 10 worst affected countries", # the axis names

    color_discrete_sequence=["pink"], 

    height=500,

    width=800

)
world_map = folium.Map(location=[11,0], tiles="cartodbpositron", zoom_start=2, max_zoom = 6, min_zoom = 2)





for i in range(0,len(confirmed_df)):

    folium.Circle(

        location=[confirmed_df.iloc[i]['lat'], confirmed_df.iloc[i]['long']],

        fill=True,

        radius=(int((np.log(confirmed_df.iloc[i,-1]+1.00001)))+0.2)*50000,

        color='red',

        fill_color='indigo',

        tooltip = "<div style='margin: 0; background-color: black; color: white;'>"+

                    "<h4 style='text-align:center;font-weight: bold'>"+confirmed_df.iloc[i]['country'] + "</h4>"

                    "<hr style='margin:10px;color: white;'>"+

                    "<ul style='color: white;;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

                        "<li>Confirmed: "+str(confirmed_df.iloc[i,-1])+"</li>"+

                        "<li>Deaths:   "+str(death_df.iloc[i,-1])+"</li>"+

                        "<li>Death Rate: "+ str(np.round(death_df.iloc[i,-1]/(confirmed_df.iloc[i,-1]+1.00001)*100,2))+ "</li>"+

                    "</ul></div>",

        ).add_to(world_map)



world_map