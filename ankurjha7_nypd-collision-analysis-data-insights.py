# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.offline as offline
from plotly.offline import iplot
import plotly.figure_factory as ff
plotly.offline.init_notebook_mode()
import dateutil

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
#Loading data and parsing date if in string 
df_raw = pd.read_csv("../input/nypd-collision/NYPD_Motor_Vehicle_Collisions.csv",low_memory=False)
df_raw['DATE'] = pd.to_datetime(df_raw['DATE'])
def get_plot_fig_data(df,x_field_name,y_field_name):
    data = go.Scatter(
          x=df[x_field_name],
          y=df[y_field_name],
    mode = 'lines',
    name = y_field_name)
    return data
def get_plot_layout(title,x_axis_name,y_axis_name):
    # specify the layout of our figure
    layout = go.Layout(
    title=title,
    xaxis=dict(
        title=x_axis_name,
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title=y_axis_name,
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
    return layout
#Method to plot time based distribution 
def get_date_based_count(df,count_index_name,range = '1M'):
    df_raw_grouped = df.groupby(pd.Grouper(key='DATE',freq = range )).size().reset_index(name = count_index_name)
    return df_raw_grouped
def get_date_based_sum(df,field_names = [],range = '1M'):
    df_raw_grouped = df.groupby(pd.Grouper(key='DATE',freq = range )).sum()[field_names].reset_index()
    return df_raw_grouped
#Distribution of Accidents based on datetime
data = get_date_based_count(df_raw,count_index_name="No_Of_Accidents",range='1M')
fig = go.Figure(data=[get_plot_fig_data(data,'DATE',"No_Of_Accidents")], layout=get_plot_layout("Number  Of Accidents","DATE",""))
iplot(fig)
#distribution of pedestrian casuality on the basis of date(and time is required)
data = get_date_based_sum(df_raw,field_names=['NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED'],range='1M')
data['PEDESTRIAN_CASUALITY'] = data['NUMBER OF PEDESTRIANS INJURED'] + data['NUMBER OF PEDESTRIANS KILLED']

fig = go.Figure(data=[get_plot_fig_data(data,'DATE',"PEDESTRIAN_CASUALITY")], layout=get_plot_layout("Number  Of Pedestrian Casuality","DATE",""))
iplot(fig)

#Distribution of cyclists' casuality on the basis of datetime
data = get_date_based_sum(df_raw,field_names=['NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED'],range='1M')
data['CYCLIST_CASUALITY'] = data['NUMBER OF CYCLIST INJURED'] + data['NUMBER OF CYCLIST KILLED']

fig = go.Figure(data=[get_plot_fig_data(data,'DATE',"CYCLIST_CASUALITY")], layout=get_plot_layout("Number  Of Cyclist Casuality","DATE",""))
iplot(fig)
#Distribution of casuality to per accident ration on the basis of time 
# df_raw['TOTAL_CASUALITIES'] = df_raw['']+ 
data_accidents = get_date_based_count(df_raw,count_index_name="No_Of_Accidents",range='1M')
data_casualities = get_date_based_sum(df_raw,field_names=['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED'],range='1M')
result = pd.merge(data_accidents,data_casualities,on='DATE')

trace_accidents = get_plot_fig_data(result,'DATE',"No_Of_Accidents")
trace_injured =  get_plot_fig_data(result,'DATE',"NUMBER OF PERSONS INJURED")
trace_killed = get_plot_fig_data(result,'DATE',"NUMBER OF PERSONS KILLED")

fig = go.Figure(data=[trace_accidents,trace_injured,trace_killed],layout=get_plot_layout("Accident and Casualities","DATE",""))
iplot(fig)




##The basic code snippet has been developed with the help of - https://plot.ly/~empet/14692/mapbox-choropleth-that-works-with-plotly/#/

##creating dictionary mapping borough to total number of accidents 
df_raw_grouped = df_raw.groupby(by='BOROUGH').size().reset_index(name = "count_of_accidents")
borough_accident_dic = dict()
for index,rows in df_raw_grouped.iterrows():
    borough_accident_dic.update({rows['BOROUGH'] : rows['count_of_accidents']})
    
##plotting Newyork map using geojson
geoJSON = pd.read_json("../input/newyorkcityborough/newyork-city-json.json")
boroughs=[geoJSON['features'][k]['properties']['borough'] for k in range(len(geoJSON['features']))]
accidents=[borough_accident_dic[borough.upper()] for borough in boroughs]

###setting colors for each borough
colors = ['rgb(255, 255, 204)',
          'rgb(161, 218, 180)',
          'rgb(65, 182, 196)',
          'rgb(44, 127, 184)',
          'rgb(8, 104, 172)',
         'rgb(37, 52, 148)']

facecolor = [ colors[accident%len(colors)] for accident in accidents]



##Setting latitude a longitude for finding boundaries.Chosing a central location for each county
lons=[]
lats=[]
for k in range(len(geoJSON['features'])):
    county_coords=np.array(geoJSON['features'][k]['geometry']['coordinates'][0])
    m, M =county_coords[:,0].min(), county_coords[:,0].max()
    lons.append(0.5*(m+M))
    m, M =county_coords[:,1].min(), county_coords[:,1].max()
    lats.append(0.5*(m+M))


##Adding Mapbox sources 
sources=[]
for feat in geoJSON['features']: 
        sources.append({"type": "FeatureCollection", 'features': [feat]})

text=[b+'<br>Number of Accidents: '+str(a) for b, a in zip(boroughs, accidents)]


NewYork = dict(type='scattermapbox',
             lat=lats, 
             lon=lons,
             mode='markers',
             text=text,
             marker=dict(size=1, color=facecolor),
             showlegend=False,
             hoverinfo='text'
            )

layers=[dict(sourcetype = 'geojson',
             source =sources[k],
             below="water", 
             type = 'fill',   
             color = facecolor[k],
             opacity=0.8
            ) for k in range(len(sources))]

layout = dict(title='Mapbox Choropleth<br>Accidents in Newyork',
              font=dict(family='Balto'),
              autosize=False,
              width=800,
              height=800,
              hovermode='closest',
              mapbox=dict(accesstoken="pk.eyJ1IjoiYW5rdXJqaGE3IiwiYSI6ImNqcTgyczgzYzA4azM0OXBmZWN4a3dqeTAifQ.Be0NnPr21arvrvE2Z-5DCA",
                          layers=layers,
                          bearing=0,
                          center=dict(
                          lat=40.721319, 
                          lon=-73.987130),
                          pitch=0,
                          zoom=8,
                    ) 
              )

fig = dict(data=[NewYork], layout=layout)
iplot(fig)