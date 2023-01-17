#Import all the required libraries

#Graphical Libraries

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import folium

from folium.plugins import MarkerCluster, MiniMap, Fullscreen

import branca

from IPython.display import IFrame, YouTubeVideo



#Manipulation

from datetime import date

import pandas as pd

import numpy as np



#Kaggle default

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#To supress warnings

import warnings

warnings.filterwarnings('ignore')



#To get the geolocation details

from geopy.extra.rate_limiter import RateLimiter

from geopy.geocoders import Nominatim
from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('mOV1aBVYKGA',width=600, height=400)
#Import the dataset

df=pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")



#Verify the first five rows for the sanity check

df.head()
#Verify the shape of the data

print("Shape of dataframe: ", df.shape)
#Convert the columns from float to int, and respective date columns for further analysis

df = df.astype({'Confirmed': 'int32', 'Deaths': 'int32', 

                'Recovered': 'int32', 'Last Update': 'datetime64',

                'Date': 'datetime64'})

df['Country'] = df['Country'].replace({'Mainland China': 'China'})
#Get the data of the latest date from the dataset

maxDate = max(df['Date'])

df_lastDate = df[df['Date'] >  pd.Timestamp(date(maxDate.year,maxDate.month,maxDate.day))]
#Print the total number of observations on cases -Worldwide

print('\033[1mTotal Confirmed cases worldwide: ',df_lastDate.Confirmed.sum())

print('\033[1mTotal Death cases worldwide: ',df_lastDate.Deaths.sum())

print('\033[1mTotal Recovered cases worldwide: ',df_lastDate.Recovered.sum())
#Process data for each country

df_tempC = df_lastDate.groupby('Country').Confirmed.sum().to_frame()

df_tempD = df_lastDate.groupby('Country').Deaths.sum().to_frame()

df_tempR = df_lastDate.groupby('Country').Recovered.sum().to_frame()



#Merge the above data frames into one for convenient processing

df_temp = pd.merge(df_tempC, df_tempD, how='inner', left_index=True, right_index=True)

df_temp = pd.merge(df_temp, df_tempR, how='inner', left_index=True, right_index=True)

df_temp = df_temp.sort_values(['Confirmed'],ascending=[False])





#Create an interactive table based and fill the final data frame values

fig = go.Figure(data=[go.Table(header=dict(values=['<b>Country</b>','<b>Confirmed Cases</b>',

                                                   '<b>Death Cases</b>', '<b>Recovered Cases</b>'],

                                            fill_color='paleturquoise', 

                                            align=['left','center'],

                                           font=dict(color='black', size=16),

                                           height=40),

                               cells=dict(values=[df_temp.index, 

                                                  df_temp.Confirmed.values, df_temp.Deaths.values,

                                                 df_temp.Recovered.values,],

                                          fill_color='lavender', 

                                          align=['left','center'],

                                          font=dict(color='black', size=14),

                                         height=23)



                                )

                     ]

               )



#Cosmetic changes

fig.update_layout(title={'text':'<b>Number of Confirmed, Death and Recovered cases in each country</b>',

                        'y':0.92,

                        'x':0.5,

                        'xanchor':'center',

                        'yanchor':'top',

                        'font':dict(size=22)

                        }

                 )

fig.update_layout(height=600)
print("\033[1mTotal number of countries affected:", len(df_temp.Confirmed.values))
#Process cases of China and Rest of the World

chinaConfirmed = df_lastDate[df_lastDate.Country=='China'].Confirmed.sum()

notChinaConfirmed = df_lastDate[df_lastDate.Country!='China'].Confirmed.sum()

chinaDeaths = df_lastDate[df_lastDate.Country=='China'].Deaths.sum()

notChinaDeaths = df_lastDate[df_lastDate.Country!='China'].Deaths.sum()

chinaRecovered = df_lastDate[df_lastDate.Country=='China'].Recovered.sum()

notChinaRecovered = df_lastDate[df_lastDate.Country!='China'].Recovered.sum()



#yAxis labels for the figure

yAxisChina = [chinaConfirmed, chinaDeaths, chinaRecovered]

yAxisNotChina = [notChinaConfirmed,notChinaDeaths,notChinaRecovered]



x=['Confirmed', 'Death', 'Recovered']

fig = go.Figure(go.Bar(x=x, y=[chinaConfirmed, chinaDeaths, chinaRecovered],text=yAxisChina, textposition='outside',

                       hovertemplate = "%{x}: %{y} </br>", name='China', marker_color='rgb(55, 83, 109)'))

fig.add_trace(go.Bar(x=x, y=[notChinaConfirmed,notChinaDeaths,notChinaRecovered],text=yAxisNotChina, textposition='outside', 

                     hovertemplate = "%{x}: %{y} </br>", name='Rest of the World',marker_color='rgb(26, 118, 255)'))



fig.update_layout(barmode='group', xaxis={'categoryorder':'category ascending'})

fig.update_layout(

    title={'text':'<b>Number of Confirmed, Death and Recovered cases in China and Rest of the World</b>',

                       'x':0.1,'xanchor':'left','font':dict(size=20,color='black')},

    xaxis_tickfont_size=14,

    legend=dict(

        x=1,

        y=1,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15 # gap between bars of adjacent location coordinates.

#     bargroupgap=0.1 # gap between bars of the same location coordinate.

)



# Update xaxis properties

fig.update_xaxes(title_text="Type of Cases",  titlefont_size=16, tickfont_size=15)



# Update yaxis properties

fig.update_yaxes(title_text="Number of Cases", titlefont_size=16, tickfont_size=15)



fig.show()
#Process and Merge the location details

df_lastDate['Province/State Copy'] = df_lastDate['Province/State'].fillna(' ')



df_lastDate['fullAddress'] = np.where((df_lastDate['Province/State Copy']==' ') |

                                         (df_lastDate['Province/State Copy']==df_lastDate['Country']), 

                                      df_lastDate['Country'], 

                                      df_lastDate['Province/State Copy'] + ', ' + df_lastDate['Country'])
locator = Nominatim(user_agent="myGeocoder")

#put a delay of 1 second and fetch the geolocation details of each location

geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

df_lastDate['location'] = df_lastDate['fullAddress'].apply(geocode)

df_lastDate['point'] = df_lastDate['location'].apply(lambda loc: tuple(loc.point) if loc else None)

df_lastDate[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df_lastDate['point'].tolist(), index=df_lastDate.index)
#Initialize map

intializeMap = folium.Figure(height=500)

custom_map = folium.Map(location=[42, 12], zoom_start=2, tiles='cartodbpositron').add_to(intializeMap)



fullscreen= Fullscreen(position='topright', 

                       title='Fullscreen', 

                       title_cancel='Exit Fullscreen', 

                       force_separate_button=True

                      ).add_to(custom_map)



#Create a custom html to show legend on map

legend_html = '''

                    {% macro html(this, kwargs) %}

                    <div style="

                        position: fixed; 

                        bottom: 50px;

                        left: 50px;

                        width: 250px;

                        height: 80px;

                        z-index:9999;

                        font-size:14px;

                        ">

                        <p>&emsp;     <i class="fa fa-map-marker fa-2x" style="color:darkblue"></i> Confirmed cases</p>

                        <p>&emsp;     <i class="fa fa-map-marker fa-2x" style="color:red"></i> Confirmed Deaths cases</p>

                    </div>

                    <div style="

                        position: fixed; 

                        bottom: 50px;

                        left: 50px;

                        width: 150px;

                        height: 80px; 

                        z-index:9998;

                        font-size:14px;

                        background-color: #ffffff;

                        filter: blur(8px);

                        -webkit-filter: blur(8px);

                        opacity: 0.7;

                        ">

                    </div>

                    {% endmacro %}

                '''

legend = branca.element.MacroElement()

legend._template = branca.element.Template(legend_html)

custom_map.get_root().add_child(legend)



#Add locations and styling on map

df_lastDate.apply(lambda row: folium.Marker(location=[row["latitude"], row["longitude"]]

                                                , popup = ('<b>Province/Country:</b> ' + row["fullAddress"]

                                                          +'<br>'+'<b>Confirmed:</b> ' + str(row['Confirmed'])

                                                          +'<br>'+'<b>Deaths:</b> ' + str(row['Deaths'])

                                                          +'<br>'+'<b>Recovered:</b> ' + str(row['Recovered'])

                                                          )

                                                , icon=folium.Icon(color='darkblue',icon='info-sign')

                                                  ,color='rgb(55, 83, 109)',fill_color='rgb(55, 83, 109)'

                                                ).add_to(custom_map) if (row['Deaths']==0) else

                  folium.Marker(location=[row["latitude"], row["longitude"]]

                                                , popup = ('<b>Province/Country:</b> ' + row["fullAddress"]

                                                          +'<br>'+'<b>Confirmed:</b> ' + str(row['Confirmed'])

                                                          +'<br>'+'<b>Deaths:</b> ' + str(row['Deaths'])

                                                          +'<br>'+'<b>Recovered:</b> ' + str(row['Recovered'])

                                                          )

                                                  ,icon=folium.Icon(color='red', icon='info-sign')

                                                  ,color='rgb(26, 118, 255)',fill_color='rgb(26, 118, 255)'

                                                ).add_to(custom_map), axis=1)



custom_map
#Process data of Chinese provinces

chinaConfirmed = df_lastDate[df_lastDate.Country=='China'].Confirmed[:10]

chinaDeath = df_lastDate[df_lastDate.Country=='China'].Deaths[:10]

chinaRecovered = df_lastDate[df_lastDate.Country=='China'].Recovered[:10]

chinaProvinceName = df_lastDate[df_lastDate.Country=='China']['Province/State'][:10]



#Initialize the figure and start adding the traces

#China Confirmed cases

fig = go.Figure()

fig.add_trace(go.Bar(

                    y=chinaProvinceName,

                    x=chinaConfirmed,

                    name='Confirmed',

                    hovertemplate = "%{x}: %{y} </br>",

                    orientation='h',

                    marker=dict(

                                color='yellow',

                                line=dict(color='yellow', width=3)

                                )

                    )

             )

#China Death cases

fig.add_trace(go.Bar(

                    y=chinaProvinceName,

                    x=chinaDeath,

                    name='Death',

                    hovertemplate = "%{x}: %{y} </br>",

                    orientation='h',

                    marker=dict(

                        color='red',

                        line=dict(color='red', width=3)

                                )

                    )

             )

#China Recovered cases

fig.add_trace(go.Bar(

                    y=chinaProvinceName,

                    x=chinaRecovered,

                    name='Recovered',

                    hovertemplate = "%{x}: %{y} </br>",

                    orientation='h',

                    marker=dict(

                        color='green',

                        line=dict(color='green', width=3)

                                )

                    )

             )



#Cosmetic changes to figure

fig.update_layout(

                    title={'text':'<b>Top 10 provinces of China having highest number of Corona Virus cases</b>',

                           'x':0.15,'xanchor':'left','font':dict(size=20,color='black')

                          }

                 )

fig.update_layout(legend_orientation="h", 

                  legend=dict(

                                x=0.25,

                                y=-0.2,

                                bgcolor='rgba(255, 255, 255, 0)',

                                bordercolor='red'

                             )

                 )

# Update xaxis properties

fig.update_xaxes(title_text="Number of cases",  titlefont_size=18, tickfont_size=15)



# Update yaxis properties

fig.update_yaxes(title_text="China Provinces", titlefont_size=18, tickfont_size=15)

fig.update_layout(barmode='stack', height=600)

fig.show()
#Process the data of China based grouped by date

chinaTimelineC = df[df['Country']=='China'].groupby(df['Date'].dt.date)['Confirmed'].sum()

chinaTimelineD = df[df['Country']=='China'].groupby(df['Date'].dt.date)['Deaths'].sum()

chinaTimelineR = df[df['Country']=='China'].groupby(df['Date'].dt.date)['Recovered'].sum()



#Create figure with subplots

fig = make_subplots(rows=1, cols=2,  vertical_spacing=0.1, subplot_titles=("Confirmed Cases", "Death and Recovered Cases"))



#China confirmed cases

fig.add_trace(go.Scatter(name='Confirmed Cases',

                        y=chinaTimelineC.values,

                        x=chinaTimelineC.index,

                        text=chinaTimelineC.values,

                        textposition="top center",

                        mode='lines+markers',

                        hovertemplate = "%{x}: %{y} </br>",

                        marker=dict(color='yellow', size=10, line=dict(color='rgb(55, 83, 109)', width=3)),

                        line=dict(color='rgb(55, 83, 109)', width=4)

                        ), row=1, col=1

             )



#China death cases

fig.add_trace(go.Scatter(name='Death Cases',

                        y=chinaTimelineD.values,

                        x=chinaTimelineD.index,

                        text=chinaTimelineD.values,

                        textposition="bottom right",

                        hovertemplate = "%{x}: %{y} </br>",   

                        mode='lines+markers',

                        marker=dict(color='red', size=10, line=dict(color='rgb(55, 83, 109)', width=3)),

                        line=dict(color='rgb(55, 83, 109)', width=4)

                        ), row=1, col=2

             )



#China recovered cases

fig.add_trace(go.Scatter(name='Recovered Cases',

                        y=chinaTimelineR.values,

                        x=chinaTimelineR.index,

                        text=chinaTimelineR.values,

                        textposition="bottom right",

                        hovertemplate = "%{x}: %{y} </br>",

                        mode='lines+markers',

                        marker=dict(color='rgb(0, 196, 0)', size=10, line=dict(color='rgb(55, 83, 109)', width=3)),

                        line=dict(color='rgb(55, 83, 109)', width=4)

                        ), row=1, col=2

             )



#Cosmetic changes to figure

fig.update_layout(

                    title={'text':'<b>Comparision of Confirmed with Death and Recovered cases in China</b>',

                            'x':0.2,'xanchor':'left','font':dict(size=20,color='black')

                          }

                 )



fig.update_layout(legend_orientation="h", 

                  legend=dict(

                                x=0.25,

                                y=-0.3,

                                bgcolor='rgba(255, 255, 255, 0)',

                                bordercolor='red'

                            )

                 )



# Update xaxis properties

fig.update_xaxes(title_text="Timeline",  titlefont_size=16, tickfont_size=15, row=1, col=1)

fig.update_xaxes(title_text="Timeline",  titlefont_size=16, tickfont_size=15, row=1, col=2)



# Update yaxis properties

fig.update_yaxes(title_text="Number of Cases", titlefont_size=16, tickfont_size=15, row=1, col=1)

fig.update_yaxes(title_text="Number of Cases", titlefont_size=16, tickfont_size=15, row=1, col=2)

fig.show()
#Process the data of Rest of the world grouped by date

notChinaTimelineC = df[df['Country']!='China'].groupby(df['Date'].dt.date)['Confirmed'].sum()

notChinaTimelineD = df[df['Country']!='China'].groupby(df['Date'].dt.date)['Deaths'].sum()

notChinaTimelineR = df[df['Country']!='China'].groupby(df['Date'].dt.date)['Recovered'].sum()



#Create figure with subplots

fig = make_subplots(rows=1, cols=2,  vertical_spacing=0.1, subplot_titles=("Confirmed Cases", "Death and Recovered Cases"))



#Rest of the World confirmed cases

fig.add_trace(go.Scatter(name='Confirmed Cases',

                        y=notChinaTimelineC.values,

                        x=notChinaTimelineC.index,

                        text=notChinaTimelineC.values,

                        textposition="top center",

                        mode='lines+markers',

                        hovertemplate = "%{x}: %{y} </br>",

                        marker=dict(color='yellow', size=10, line=dict(color='rgb(55, 83, 109)', width=3)),

                        line=dict(color='rgb(26, 118, 255)', width=4)

                        ), row=1, col=1

             )



#Rest of the World death cases

fig.add_trace(go.Scatter(name='Death Cases',

                        y=notChinaTimelineD.values,

                        x=notChinaTimelineD.index,

                        text=notChinaTimelineD.values,

                        textposition="bottom right",

                        mode='lines+markers',

                        hovertemplate = "%{x}: %{y} </br>",

                        marker=dict(color='red', size=10, line=dict(color='rgb(55, 83, 109)', width=3)),

                        line=dict(color='rgb(26, 118, 255)', width=4)

                        ), row=1, col=2

             )



#Rest of the World recovered cases

fig.add_trace(go.Scatter(name='Recovered Cases',

                        y=notChinaTimelineR.values,

                        x=notChinaTimelineR.index,

                        text=notChinaTimelineR.values,

                        textposition="bottom right",

                        mode='lines+markers',

                        hovertemplate = "%{x}: %{y} </br>",

                        marker=dict(color='rgb(0, 196, 0)', size=10, line=dict(color='rgb(55, 83, 109)', width=3)),

                        line=dict(color='rgb(26, 118, 255)', width=4)

                        ), row=1, col=2

             )



#Cosmetic changes to figure

fig.update_layout(

                title={'text':'<b>Comparision of Confirmed  cases v/s Death and Recovered cases globally excluding China</b>',

                            'x':0.09,'xanchor':'left','font':dict(size=20,color='black')

                       }

                )

fig.update_layout(legend_orientation="h", 

                  legend=dict(

                                x=0.25,

                                y=-0.3,

                                bgcolor='rgba(255, 255, 255, 0)',

                                bordercolor='red'

                             )

                 )

# Update xaxis properties

fig.update_xaxes(title_text="Timeline",  titlefont_size=16, tickfont_size=15, row=1, col=1)

fig.update_xaxes(title_text="Timeline",  titlefont_size=16, tickfont_size=15, row=1, col=2)



# Update yaxis properties

fig.update_yaxes(title_text="Number of Cases", titlefont_size=16, tickfont_size=15, row=1, col=1)

fig.update_yaxes(title_text="Number of Cases", titlefont_size=16, tickfont_size=15, row=1, col=2)

fig.show()