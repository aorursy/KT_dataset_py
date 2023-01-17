import warnings

warnings.filterwarnings('ignore')
import pandas as pd

oly_evts = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv", header = 'infer')

oly_rgn = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv", header='infer')
print(oly_evts.shape)

print(oly_rgn.shape)

oly_evts.head()
#Missing values with respect to each column in the dataset

import seaborn as sns

sns.heatmap(oly_evts.isnull(), cbar=False)
#Missing value correlation

import missingno as msno

msno.heatmap(oly_evts)
import numpy as np

imp_col = ['Age', 'Height', 'Weight']

for col in imp_col:

    oly_evts[col] = oly_evts[col].fillna(np.mean(oly_evts[col]))

    oly_evts[col] = np.round(oly_evts[col],1)
from pandasql import sqldf

m_vs_fm_120 = """select Year, Sex, count(distinct Name) cnt from oly_evts group by Year, Sex"""

pysqldf = lambda m_vs_fm: sqldf(m_vs_fm, globals())

mvf_120_df = pysqldf(m_vs_fm_120)



m_120_df = mvf_120_df[mvf_120_df.Sex=='M']

f_120_df = mvf_120_df[mvf_120_df.Sex=='F']
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



m_120_t = go.Bar(

    x=m_120_df.Year,

    y=m_120_df.cnt,

    name='Male'

) 



f_120_t = go.Bar(

    x=f_120_df.Year,

    y=f_120_df.cnt,

    name='Female'

) 



data = [m_120_t, f_120_t]

layout = go.Layout(

    barmode='stack', 

    xaxis=dict(type='category', title='Year'),

    yaxis=dict(title='Count of Players'),

    title="Male Vs Female Participation in Olympic from the year 1896 - 2016"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='stack-bar')
from pandasql import sqldf

na_120 = """select Year, count(distinct NOC) cnt from oly_evts group by Year"""

pysqldf = lambda m_vs_fm: sqldf(na_120, globals())

na_120_df = pysqldf(na_120)
import plotly.plotly as py

import plotly.graph_objs as go

import  colorlover as cl

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



na_120_t = go.Bar(

    x=na_120_df.Year,

    y=na_120_df.cnt,

    marker=dict(

        color='rgb(163,132,193)',

        line=dict(

            color='rgb(239,234,244)',

            width=2)

    )

)



data = [na_120_t]

layout = go.Layout(

    barmode='stack', 

    xaxis=dict(type='category', title='Year'),

    yaxis=dict(title='Count of Nations'),

    title="Nations participating in Olympic from the year 1896 - 2016"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='stack-bar')
from pandasql import sqldf

Medals_120 = """select Year, Sex, Medal, count(Medal) cnt from oly_evts group by Year, Sex, Medal"""

pysqldf = lambda m_vs_fm: sqldf(Medals_120, globals())

Medals_120_df = pysqldf(Medals_120)

na_key = Medals_120_df["Medal"].isnull()

Medals_120_df_final = Medals_120_df.loc[~na_key]
import plotly.plotly as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



medal_t = ff.create_facet_grid(

    Medals_120_df_final,

    x='Year',

    y='cnt',

    color_name='Medal',

    facet_row='Medal',

    facet_col='Sex',

    show_boxes=False,

    colormap={'Gold': 'rgb(218,165,32)', 'Silver': 'rgb(211,211,211)', 'Bronze': 'rgb(128, 64, 0)'}

)



medal_t.layout.update({'title': 'Male Vs Female wining medals in Olympic from the year 1896 - 2016'})



iplot(medal_t, filename='Male Vs Female wining medals in Olympic from the year 1896 - 2016')
from pandasql import sqldf

loc_120 = """select distinct Year, Season, City, count(distinct Sport) cnt from oly_evts group by Year, Season, City"""

pysqldf = lambda m_vs_fm: sqldf(loc_120, globals())

loc_120_df = pysqldf(loc_120)
#Summer Olympics

sum_df = loc_120_df[loc_120_df.Season == 'Summer']

#Winter Olympics

win_df = loc_120_df[loc_120_df.Season == 'Winter']
#Location Details of Cities

loc_det = pd.DataFrame({

        'City':['Barcelona', 'London', 'Antwerpen', 'Paris', 'Calgary', 'Albertville',

           'Lillehammer', 'Los Angeles', 'Salt Lake City', 'Helsinki', 'Lake Placid',

           'Sydney', 'Atlanta', 'Stockholm', 'Sochi', 'Nagano', 'Torino', 'Beijing',

           'Rio de Janeiro', 'Athina', 'Squaw Valley', 'Innsbruck', 'Sarajevo',

           'Mexico City', 'Munich', 'Seoul', 'Berlin', 'Oslo', "Cortina d'Ampezzo",

           'Melbourne', 'Roma', 'Amsterdam', 'Montreal', 'Moskva', 'Tokyo', 'Vancouver',

           'Grenoble', 'Sapporo', 'Chamonix', 'St. Louis', 'Sankt Moritz', 'Garmisch-Partenkirchen'],

    'lat':[41.38, 51.50, 51.22, 48.85, 51.04, 34.26, 61.115, 34.05, 40.76,

          60.16, 27.29, -33.86, 33.74, 59.32, 43.58, 36.65, 45.06, 39.90,

          -22.911, 37.97, 36.72, 47.26, 43.85, 19.43, 48.13, 37.56, 52.52, 

          59.91, 46.53, -37.81, 41.89, 52.37, 45.50, 44.81, 35.65, 49.26,

          45.18, 43.01, 45.92, 38.63, 46.49, 47.5],

    'lon':[2.173, -0.12, 4.39, 2.35, -114.07, -86.20, 10.46, -118.24, -111.89,

          24.93, -81.36, 151.20, -84.38, 18.06, 39.72, 138.18, 7.68, 116.40,

          -43.2094, 23.73, -119.23, 11.39, 18.41, -99.13, 11.58, 126.97, 13.40,

          10.75, 12.13, 144.96, 12.48, 4.9, -73.55, 20.46, 139.74, -123.11,

          5.72, 141.40, 6.86, -90.19, 9.83, 11.08],



})
#Merging Summer and Winter dataset with loc_det

sum_df = pd.merge(sum_df, loc_det, on='City')

win_df = pd.merge(win_df, loc_det, on='City')
#Locations of Summer Olympics from 1896 to 2016

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

plt.figure(figsize=(25,15))



map = Basemap()

map.drawcoastlines()

map.drawcountries()

map.fillcontinents()

map.drawmapboundary()



map.drawmapboundary(fill_color='k')

map.fillcontinents(color='silver',lake_color='k')

 

lons = pd.np.array(sum_df['lon'])

lats = pd.np.array(sum_df['lat'])



x,y = map(lons, lats)



#map.scatter(x, y, marker='D',color='m')



map.plot(x, y, marker='+', color='m', markersize=6, markeredgewidth=2)



plt.title("Locations of Summer Olympics from 1896 to 2016")

plt.show()
#Locations of Winter Olympics from 1896 to 2016

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

plt.figure(figsize=(25,15))



map = Basemap()

map.drawcoastlines()

map.drawcountries()

map.fillcontinents()

map.drawmapboundary()



map.drawmapboundary(fill_color='k')

map.fillcontinents(color='silver',lake_color='k')

 

lons = pd.np.array(win_df['lon'])

lats = pd.np.array(win_df['lat'])



x,y = map(lons, lats)



#map.scatter(x, y, marker='D',color='m')



map.plot(x, y, marker='+', color='m', markersize=6, markeredgewidth=2)



plt.title("Locations of Winter Olympics from 1896 to 2016")

plt.show()