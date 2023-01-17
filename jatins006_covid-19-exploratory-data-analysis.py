#Installations
!pip install calmap
!pip install chart_studio
!pip install plotly-geo

import chart_studio
#API KEY 
username = 'jatins' # your username
api_key = 'ZeezwRMdl79LdkA45Tcy'
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
# W1Qor7JU3YzGZkHwqCvT
#Import necessary packages

import numpy as np # linear algebra
import pandas as pd # data processing
import requests
import json

#Visualization Libraries

import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import calmap


from IPython.core.display import HTML

%matplotlib inline
pio.templates.default = "plotly_dark"
#New York Times GitHub Data
nyturl = 'https://api.github.com/repos/nytimes/covid-19-data/contents'
r = requests.get(nyturl)
if(r.ok):
    repo = json.loads(r.text or r.content)
    
repo[2]

d = {}

for i in range(len(repo)):
    htmlurl = repo[i]['download_url']
    name = repo[i]['name']
    if name.find(".csv") != -1:
        tempdf = pd.read_csv(htmlurl)
        d["df_" + name] = tempdf
        
d.keys()


#Read data
df = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
US = pd.read_csv("../input/corona-virus-report/usa_county_wise.csv")
nyt_us_counties = d['df_us-counties.csv']
nyt_us_states = d['df_us-states.csv']
# #Weather history test code
# !pip install wwo-hist
# from wwo_hist import retrieve_hist_data

# frequency=3
# start_date = '11-DEC-2018'
# end_date = '11-MAR-2019'
# api_key = '5601834ed1f74cddb2402604201903'
# location_list = ['singapore','california']

# hist_weather_data = retrieve_hist_data(api_key,
#                                 location_list,
#                                 start_date,
#                                 end_date,
#                                 frequency,
#                                 location_label = False,
#                                 export_csv = False,
#                                 store_df = True)


# singaporetemp = hist_weather_data[0]
# singaporetemp.head()
# singaporetemp[['date_time', 'maxtempC']].groupby('date_time').sum()

# import seaborn as sns
# sns.scatterplot(x= 'date_time', y= 'maxtempC', data = singaporetemp)
# #Library - Pandas Profiling 
# import pandas_profiling as pp

# profile = pp.ProfileReport(df,title='Pandas Profiling Report', html={'style':{'full_width':True}})
# profile

#Clean Data & Change data types 

df['Date'] = pd.to_datetime(df['Date'])
US['Date'] = pd.to_datetime(US['Date'])

#Create Variables

df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
US['Active'] = US['Confirmed'] - US['Deaths']

US.info()
#Number for Countries affected over time

dfpivot = df.pivot_table(index = 'Country/Region', columns = 'Date', values = ['Confirmed'])

countrycount =  dfpivot.groupby("Country/Region").sum().apply(lambda x: x[x > 0].count(), axis =0)
countrycount = pd.DataFrame(countrycount)

countrycount.reset_index(inplace = True)
countrycount.columns = ["Metric", "Date", "Count"]


#Animated Line Chart
trace = go.Scatter(x=countrycount['Date'][0:2], y=countrycount['Count'][0:2],
                         mode = 'markers', line = dict(width = 2))

frames = [dict(data = [dict(type = 'scatter', x= countrycount['Date'][:k+1], y = countrycount['Count'][:k+1])],
               traces = [0,1], 
               ) for k in range(1, len(countrycount) - 1)
         ]
    
layout = go.Layout(width = 600, 
                   height = 440, 
                   showlegend = False, hovermode = 'closest', 
                    updatemenus=[dict(type='buttons', showactive=False,
                                y=1.05,
                                x=1.15,
                                xanchor='right',
                                yanchor='top',
                                pad=dict(t=0, r=10),
                                buttons=[dict(label='Play',
                                              method='animate',
                                              args=[None, 
                                                    dict(frame=dict(duration=30, 
                                                                    redraw=False),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True,
                                                         mode='immediate')])])])
      

layout.update(xaxis =dict(range=[countrycount.Date[0], countrycount.Date[len(countrycount)-1]], 
                          autorange=False, showgrid = True, showline = True,
                          showticklabels=True,
                          linecolor = 'rgb(204, 204, 204)', 
                          linewidth = 2
                         ),
              yaxis =dict(range=[min(countrycount.Count)-10, max(countrycount.Count)+20], 
                          autorange=False, showline = True,
                          showticklabels=True,
                          linecolor = 'rgb(204, 204, 204)', 
                          linewidth = 2
                         ), 
              title = "Number Of Countries Affected Over Time"
             );
fig = go.Figure(data=[trace], frames=frames, layout=layout)

fig.show()


import chart_studio.plotly as py
py.plot(fig, filename = 'countries-affected-over-time', auto_open=True)
#Race Chart
# HTML('''ources/embed.js"></script></div>''')<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1873703" data-url="https://flo.uri.sh/visualisation/1873703/embed"><script src="https://public.flourish.studio/res
#Visualize Count Data

#Get latest data
dflatest = df[df['Date'] == max(df['Date'])]

#Aggregate Values
grouped = dflatest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()
summary = grouped.sort_values(by = 'Confirmed', ascending = False).reset_index()

#Custom formatting
summary.style.background_gradient(cmap= 'Blues', subset = ['Confirmed'])\
             .background_gradient(cmap= 'Reds', subset = ['Deaths'])\
             .background_gradient(cmap= 'Greens', subset = ['Recovered'])\
             .background_gradient(cmap = 'Oranges', subset = ['Active'])
#Function for trend plot
def trendplot(data, xaxis, lables, title, line_size, mode_size, colors):
    fig = go.Figure()

    for i in range(len(labels)):
        fig.add_trace(go.Scatter(x=data[xaxis], y=data[labels[i]], mode='lines+markers',
            name=labels[i],
            line=dict(color=colors[i], width=line_size[i]),
            connectgaps=True
        ))

    fig.update_layout(
        xaxis = dict(
                showline = True, 
                showgrid = True, 
                showticklabels = True,
                linecolor = 'rgb(204, 204, 204)', 
                linewidth = 2, 
        ticks = 'inside'), 

        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=True,
            linecolor = 'rgb(204, 204, 204)', 
            linewidth = 2
        ),

        autosize = True, 
        title = title, 
        hovermode = 'x'

    )
    fig.show()
    return fig


#World Trend Plot
worldsummary = df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index() 

xaxis = 'Date'
title = 'World Trend'
labels = ['Confirmed', 'Deaths', 'Recovered', 'Active']
colors = ['#51C1F9', '#FF4B4B', '#2ECC71', '#ECF0F1']
mode_size = [1,1,1,1]
line_size = [3,3,3,3]  

figtrend = trendplot(data = worldsummary, xaxis = xaxis, 
          lables = labels, title = title, 
          line_size = line_size, mode_size = mode_size, 
          colors = colors)

import plotly.io as pio
pio.write_html(figtrend, file="worldtrend.html", auto_open=True)
# World Folium Map

world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2,max_zoom=6,min_zoom=2)
for i in range(0,len(dflatest)):
    folium.Circle(
        location=[dflatest.iloc[i]['Lat'], dflatest.iloc[i]['Long']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+dflatest.iloc[i]['Country/Region']+"</h5>"+
                    "<div style='text-align:center;'>"+str(np.nan_to_num(dflatest.iloc[i]['Province/State']))+"</div>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
        "<li>Confirmed: "+str(dflatest.iloc[i]['Confirmed'])+"</li>"+
        "<li>Deaths:   "+str(dflatest.iloc[i]['Deaths'])+"</li>"+
        "</ul>"
        ,
        radius= np.log(dflatest.iloc[i]['Confirmed']+1.001)*50000,
        color='#ff6600',
        fill_color='#ff8533',
        fill_opacity = 0.1,
        fill=True).add_to(world_map)

world_map


np.log(dflatest['Confirmed']+1.001)*60000
#Choropleth Map for Confirmed
pio.templates.default = "ggplot2"
fig = px.choropleth(dflatest, locations="Country/Region",
                    color=np.log10(dflatest["Confirmed"]), 
                    hover_name="Country/Region", 
                    hover_data=["Confirmed"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
                   
fig.update_layout(
#                     title=dict(text = "Interactive Confirmed Cases Heat Map (Log Scale)", 
#                             xref = 'paper'), 
#                   font = dict(size = 15, 
#                   ),
                  plot_bgcolor = '#fff', 
                  paper_bgcolor = '#fff')
fig.update_coloraxes(colorbar_title="Confirmed Cases(Log Scale)",colorscale="tealrose")
fig.update_geos(fitbounds="locations", visible=False, projection_type="orthographic", oceancolor = '#afd4db',  showocean = True, bgcolor = '#fff')
# fig.to_image("Global Heat Map confirmed.png")
fig.show()


import plotly.io as pio
pio.write_html(fig, file="confirmedcasesworldmap.html", auto_open=True)
#Choropleth Map for Deaths

fig = px.choropleth(dflatest, locations="Country/Region",
                    color=np.log10(dflatest["Deaths"]), 
                    hover_name="Country/Region", # column to add to hover information
                    hover_data=["Deaths"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
                   
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title ="Death Cases Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Death Cases(Log Scale)",colorscale="Reds")
# fig.to_image("Global Heat Map confirmed.png")
fig.show()


#Choropleth Map for Recovered

fig = px.choropleth(dflatest, locations = 'Country/Region',
                    color = np.log10(dflatest['Recovered']), 
                   hover_name = "Country/Region", 
                   hover_data = ["Recovered"], 
                   color_continuous_scale = px.colors.sequential.Plasma, locationmode = "country names")

fig.update_layout(title ="Recovered Cases Heat Map (Log Scale)" )
fig.update_geos(fitbounds = "locations", visible = False)
fig.update_coloraxes(colorbar_title="Recovered Cases(Log Scale)", colorscale = "Greens")
fig.show()
#Scatter Geo for Confirmed
pio.templates.default = "plotly_white"
data = df.groupby(['Date','Country/Region'])['Confirmed', 'Deaths'].sum().reset_index()
data["Date"] = pd.to_datetime( data["Date"]).dt.strftime('%m/%d/%Y')

fig = px.scatter_geo(data, locations = "Country/Region", locationmode = "country names", 
                     color= np.power(data["Confirmed"],0.2)- 0.5 , size= np.power(data["Confirmed"]+1,0.3)-1, hover_name="Country/Region",
                     hover_data=["Confirmed"],
                     range_color= [0, max(np.power(data["Confirmed"],0.25))], 
                     projection="natural earth", animation_frame="Date", 
                     color_continuous_scale=px.colors.sequential.Plasma,
#                      title = "Time Lapse of Confirmed Cases"
                    )

fig.update_coloraxes(colorscale = 'hot')
fig.update(layout_coloraxis_showscale = False)
fig.update_layout(plot_bgcolor = '#fff', paper_bgcolor = '#fff')
fig.update_geos(oceancolor = '#afd4db',  showocean = True, bgcolor = '#fff')
fig.show()


import plotly.io as pio
pio.write_html(fig, file="worldconfirmedtimelapse.html", auto_open=True)
#Scatter Geo for Deaths
pio.templates.default = "plotly_white"
data = df.groupby(['Date','Country/Region'])['Confirmed', 'Deaths'].sum().reset_index()
data["Date"] = pd.to_datetime( data["Date"]).dt.strftime('%m/%d/%Y')

fig = px.scatter_geo(data, locations = "Country/Region", locationmode = "country names", 
                     color= 8.8 - np.power(data["Deaths"],0.2) , size= np.power(data["Deaths"]+1,0.3)-1, hover_name="Country/Region",
                     hover_data=["Deaths"],
                     range_color= [0, max(np.power(data["Deaths"],0.2))], 
                     projection="natural earth", animation_frame="Date", 
                     color_continuous_scale=px.colors.sequential.Plasma,
                     title = "Time Lapse of Death Numbers"
                    )

fig.update_coloraxes(colorscale = 'hot')
fig.update(layout_coloraxis_showscale = False)
fig.show()



#Cal Map


# f = plt.figure(figsize=(20,10))
# f.add_subplot(2,1,1)
# calmap.yearplot(df.groupby('Date')['Confirmed'].sum().diff(), fillcolor='White', cmap='GnBu', linewidth=1,linecolor="#fafafa",year=2020,)
# plt.title("Daily Confirmed Cases",fontsize=20)
# plt.tick_params(labelsize=15)

f = plt.figure(figsize=(20,10))
f.add_subplot(2,1,1)
calmap.yearplot(df.groupby('Date')['Deaths'].sum().diff(), fillcolor='White', cmap='Reds', linewidth=1,linecolor="#fafafa",year=2020,)
plt.title("Daily Deaths",fontsize=20)
plt.tick_params(labelsize=15)

plt.savefig("calmap_deaths.png")
#Exloring US Data 
US.info()
US.head()

#Top 20 States

#Get latest data
USlatest = US[US['Date'] == max(US['Date'])]

#Aggregate Values
grouped = USlatest.groupby('Province_State')['Confirmed', 'Deaths', 'Active'].sum()
summary = grouped.sort_values(by = 'Active', ascending = False)[:20].reset_index()

#Custom formatting
summary.style.background_gradient(cmap= 'Blues', subset = ['Confirmed'])\
             .background_gradient(cmap= 'Reds', subset = ['Deaths'])\
             .background_gradient(cmap = 'Oranges', subset = ['Active'])
#Get location data for ease of respresentation

#Clean US Latest
USlatest = US[US['Date'] == max(US['Date'])]
USlatest = USlatest.dropna(subset=['FIPS'])
USlatest['FIPS'] = USlatest['FIPS'].astype(int)
USlatest['FIPS']=USlatest['FIPS'].apply(lambda x: '{0:0>5}'.format(x))

pio.templates.default = "seaborn"

import plotly.figure_factory as ff
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# fig = px.choropleth(USlatest, geojson=counties, locations="FIPS",
#                     color='Confirmed', 
#                     color_continuous_scale="RdBu",
#                     range_color=(0, 30),
#                     hover_name="Province_State", # column to add to hover information
#                     hover_data=["Confirmed"],
#                     scope="usa")
                   
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()


colorscale = ["#deecfa", "#B2C5D9", "#9EB0C4", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
    "#08519c", "#FFBBB5", "#FF5E4F"
]
endpts = list(np.linspace(0, 500, len(colorscale) - 1))
fips = USlatest['FIPS'].tolist()
values = USlatest['Confirmed'].tolist()

fig = ff.create_choropleth(
    fips=fips, values=values, scope=['usa'],
    binning_endpoints=endpts, colorscale=colorscale,
    show_state_data=True,
    show_hover=True,
    asp = 2.9,
#     title_text = 'USA by Confirmed Count',
    legend_title = 'Confirmed Count'
)
fig.layout.template = "plotly_white"
fig.update_layout(plot_bgcolor = '#fff', 
                  paper_bgcolor = '#fff', 
#                  title={
#                         'text': "USA by Confirmed Count",
#                         'xref': 'paper',
#                         'yref': 'paper',
                
                    
#                         }
                 )
fig.show()

import plotly.io as pio
pio.write_html(fig, file="usheatmap.html", auto_open=True)

#Growth for the top 10 States
pio.templates.default = "plotly_dark"

top5states = summary['Province_State'][:10].to_list()
top5states

#Plot Trend Graph for the top 5 states
ustrend =  US[US['Province_State'].isin(top5states)].groupby(['Date','Province_State'])['Confirmed', 'Deaths', 'Active'].sum().reset_index() 
#Params

fig = px.line(ustrend, x="Date", y="Active", color="Province_State",
              line_group="Province_State", hover_name="Province_State")
fig.update_layout(
        xaxis = dict(
                showline = True, 
                showgrid = True, 
                showticklabels = True,
                linecolor = 'rgb(204, 204, 204)', 
                linewidth = 2, 
        ticks = 'outside'), 

        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=True,
            showticklabels=True,
            linecolor = 'rgb(204, 204, 204)', 
            linewidth = 2
            
        ), 
    title = "Active Cases - Top 10 US States", 
    hovermode="x"
)
fig.show()

import plotly.io as pio
pio.write_html(fig, file="top10states_darkmode.html", auto_open=True)
nyt_us_counties.info()

#Change "date" field to datetime
   
nyt_us_counties['date'] = pd.to_datetime(nyt_us_counties['date'])
nyt_us_counties_recent = nyt_us_counties[nyt_us_counties.date == max(nyt_us_counties.date)]

#Top 20 Counties
top20counties = nyt_us_counties_recent.groupby(['county', 'state'])['cases', 'deaths'].sum().sort_values('cases', ascending = False)[:20].reset_index()
top20counties.style.background_gradient(cmap = 'RdGy')


pio.templates.default = "plotly_white"
temp = df
dailycases = temp[['Date', 'Country/Region', 'Confirmed']].groupby('Date')['Confirmed'].sum().diff().reset_index()

fig = px.bar(dailycases, x="Date", y="Confirmed", color_discrete_sequence=['#0f98fa'])


fig.update_layout(xaxis = dict(
                          autorange=True, showgrid = True, showline = True,
                          showticklabels=True,
                          linecolor = 'rgb(204, 204, 204)', 
                          linewidth = 2
                         ),
              yaxis =dict(
                          autorange=True, showline = True,
                          showticklabels=True,
                          linecolor = 'rgb(204, 204, 204)',
                          linewidth = 2
                         ),
              paper_bgcolor = '#fff', 
              plot_bgcolor = '#fff',
            
             );
fig.show()

import plotly.io as pio
pio.write_html(fig, file="dailycases_world.html", auto_open=True)
#Italy - Daily Cases
temp = df[df['Country/Region']=='Italy' ]
dailycases = temp.groupby('Date')['Confirmed'].sum().diff().reset_index()

fig = px.bar(dailycases, x="Date", y="Confirmed", color_discrete_sequence=['#24bfa0'])


fig.update_layout(xaxis = dict(
                          autorange=True, showgrid = True, showline = True,
                          showticklabels=True,
                          linecolor = 'rgb(204, 204, 204)', 
                          linewidth = 2
                         ),
              yaxis =dict(
                          autorange=True, showline = True,
                          showticklabels=True,
                          linecolor = 'rgb(204, 204, 204)',
                          linewidth = 2
                         ),
              paper_bgcolor = '#fff', 
              plot_bgcolor = '#fff',
            
             );
fig.show()

# import plotly.io as pio
# pio.write_html(fig, file="dailycases_world.html", auto_open=True)
def plotsubplot(data, countries, colors, ncols = 1):
    fig = make_subplots(rows = len(countries), cols = ncols, subplot_titles = countries)
    for i in range(0, len(countries)):
        temp = data[data['Country/Region']==countries[i]]
        dailycases = temp.groupby('Date')['Confirmed'].sum().diff().reset_index()
        temp_fig = px.bar(dailycases, x="Date", y="Confirmed", color_discrete_sequence=[colors[i]])
        fig.add_trace(temp_fig['data'][0], row=i+1, col=1)
    
    fig.update_layout(height=len(countries)*300,
                  title_text="Cases Per Day By Country")
    fig.show()
countries = ['India', 'US', 'Italy']

colors = ['#2a7886', '#fac248', '#4892fa']
        
plotsubplot(data = df,  countries = countries, colors = colors)