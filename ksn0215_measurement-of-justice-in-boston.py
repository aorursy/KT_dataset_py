import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import folium
# import re
plt.style.use('seaborn-notebook')
%matplotlib inline

import geopandas as gpd
import math
from shapely.geometry import Point

import plotly

plotly.tools.set_config_file(world_readable=True
                             ,sharing='public')
plotly.offline.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

from statsmodels.graphics.mosaicplot import mosaic
from IPython.display import Image
path_boston = "../input/data-science-for-good/cpe-data/Dept_11-00091/"
path_boston_acs = "../input/data-science-for-good/cpe-data/Dept_11-00091/11-00091_ACS_data/"
path_boston_shape = "../input/data-science-for-good/cpe-data/Dept_11-00091/11-00091_Shapefiles/"
boston_interview = pd.read_csv(path_boston + "11-00091_Field-Interviews_2011-2015.csv",skiprows=[1])
boston_interview.head()
import datetime
boston_interview['INCIDENT_DATE'] = pd.to_datetime(boston_interview['INCIDENT_DATE'])
boston_interview.set_index('INCIDENT_DATE', inplace=True)
len(boston_interview)
boston_interview.index.year.value_counts()
boston_interview = boston_interview['2011-01-01':'2015-12-31']
len(boston_interview)
boston_interview['2014-01-01':'2014-12-31'].head()
pd.crosstab(boston_interview["SUBJECT_RACE"], boston_interview["LOCATION_DISTRICT"],margins=True)
boston_interview_2011_race = boston_interview['2011-01-01':'2011-12-31'].groupby('SUBJECT_RACE')
boston_interview_2012_race = boston_interview['2012-01-01':'2012-12-31'].groupby('SUBJECT_RACE')
boston_interview_2013_race = boston_interview['2013-01-01':'2013-12-31'].groupby('SUBJECT_RACE')
boston_interview_2014_race = boston_interview['2014-01-01':'2014-12-31'].groupby('SUBJECT_RACE')
boston_interview_2015_race = boston_interview['2015-01-01':'2015-12-31'].groupby('SUBJECT_RACE')

df = pd.DataFrame()

df['2011'] = boston_interview_2011_race.size()
df['2012'] = boston_interview_2012_race.size()
df['2013'] = boston_interview_2013_race.size()
df['2014'] = boston_interview_2014_race.size()
df['2015'] = boston_interview_2015_race.size()

df
df = df.transpose()
df
layout = plotly.graph_objs.Layout(
    title="Occurence of Field Interviews in Boston",
    xaxis={"title":"Year"},
    yaxis={"title":"Occurence of Field Interviews"},
)


asia = go.Scatter(
                x=df.index,
                y=df['A(Asian or Pacific Islander)'],
                name = "Asian or Pacific Islander",
                line = dict(color="orange"),
                opacity = 0.8)
black = go.Scatter(
                x=df.index,
                y=df['B(Black)'],
                name = "Black",
                line = dict(color="black"),
                opacity = 0.8)
hispanic = go.Scatter(
                x=df.index,
                y=df['H(Hispanic)'],
                name = "Hispanic",
                line = dict(color="yellow"),
                opacity = 0.8)
nodata = go.Scatter(
                x=df.index,
                y=df['NO DATA ENTERED'],
                name = "No Data Entered",
                line = dict(),
                opacity = 0.8)
white = go.Scatter(
                x=df.index,
                y=df['W(White)'],
                name = "White",
                line = dict(),
                opacity = 0.8)

data= [asia,black,hispanic,nodata,white]

fig = plotly.graph_objs.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, show_link=False,config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
boston_interview['2011-01-01':'2011-12-31']["SUBJECT_RACE"].value_counts()
Image("../input/boston-racial-demographics-2010/boston_racial_demographics_2010.png")
boston_police_districts = gpd.read_file(path_boston_shape + "boston_police_districts_f55.shp")
boston_police_districts
f, ax = plt.subplots(1, figsize=(10, 8))
ax.set_axis_off()
boston_police_districts.plot(column="DISTRICT", ax=ax, cmap='tab20',edgecolor='grey',legend=True);
plt.title("Districts : Boston Police Zones")
plt.show()
boston_interview["LOCATION_CITY"].value_counts()
tidy_boston_interview = boston_interview[ ~((boston_interview['LOCATION_CITY'] == "NO DATA ENTERED") | (boston_interview['LOCATION_CITY'] == "OTHER") | (boston_interview['LOCATION_CITY'] == "Boston"))]
def func_police_districts(x):
    if x in ["Beacon Hill", "Chinatown", "South End", "North End", "Downtown"]:
        return "A1"
    elif x in ["East Boston"]:
        return "A7"
    elif x in ["Charlestown"]:
        return "A15"
    elif x in ["Roxbury", "Mission Hill"]:
        return "B2"
    elif x in ["Mattapan"]:
        return "B3"
    elif x in ["South Boston"]:
        return "C6"
    elif x in ["Dorchester"]:
        return "C11"
    elif x in ["Fenway Kenmore", "Back Bay"]:
        return "D4"
    elif x in ["Allston","Brighton"]:
        return "D14"
    elif x in ["West Roxbury"]:
        return "E5"
    elif x in ["Jamaica Plain"]:
        return "E13"
    elif x in ["Roslindale", "Hyde Park"]:
        return "E18"
    else:
        return False
tidy_boston_interview['Police District'] = tidy_boston_interview['LOCATION_CITY'].apply(func_police_districts)
tidy_boston_interview.head()
trace1 = {
  'x': tidy_boston_interview[tidy_boston_interview['SUBJECT_RACE']=='B(Black)']["Police District"].value_counts().index,
  'y': tidy_boston_interview[tidy_boston_interview['SUBJECT_RACE']=='B(Black)']["Police District"].value_counts(),
  'name': 'Black',
  'type': 'bar'
};

trace2 = {
  'x': tidy_boston_interview[tidy_boston_interview['SUBJECT_RACE']=='W(White)']["Police District"].value_counts().index,
  'y': tidy_boston_interview[tidy_boston_interview['SUBJECT_RACE']=='W(White)']["Police District"].value_counts(),
  'name': 'White',
  'type': 'bar'
};

trace3 = {
  'x': tidy_boston_interview[tidy_boston_interview['SUBJECT_RACE']=='H(Hispanic)']["Police District"].value_counts().index,
  'y': tidy_boston_interview[tidy_boston_interview['SUBJECT_RACE']=='H(Hispanic)']["Police District"].value_counts(),
  'name': 'Hispanic',
  'type': 'bar'
};

trace4 = {
  'x': tidy_boston_interview[tidy_boston_interview['SUBJECT_RACE']=='A(Asian or Pacific Islander)']["Police District"].value_counts().index,
  'y': tidy_boston_interview[tidy_boston_interview['SUBJECT_RACE']=='A(Asian or Pacific Islander)']["Police District"].value_counts(),
  'name': 'Asian or Pacific Islander',
  'type': 'bar'
};

data = [trace1,trace2,trace3,trace4]

layout = go.Layout(
    title='Frequency of Interviewed People by Race in Boston',
    barmode='relative',
    xaxis=dict(
        title='Police District'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.2,
    bargroupgap=0.1
    
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
table_interviewed = pd.crosstab(tidy_boston_interview["SUBJECT_RACE"], tidy_boston_interview["Police District"],margins=True)
table_interviewed
Image("../input/race-and-ethnicity-boston/2011-2015-Boston-in-Context-Tables_p7_filtered.png")
table_interviewed['A1']['W(White)'] / table_interviewed['A1']['All'] * 100 # White
table_interviewed['A1']['B(Black)'] / table_interviewed['A1']['All'] * 100 # Black
table_interviewed['A1']['H(Hispanic)'] / table_interviewed['A1']['All'] * 100 # Hispanic
table_interviewed['A1']['A(Asian or Pacific Islander)'] / table_interviewed['A1']['All'] * 100 # Hispanic
1123 / 57809 * 100
table_interviewed['A7']['W(White)'] / table_interviewed['A7']['All'] * 100 # White
table_interviewed['A7']['B(Black)'] / table_interviewed['A7']['All'] * 100 # Black
table_interviewed['A7']['H(Hispanic)'] / table_interviewed['A7']['All'] * 100 # Hispanic
table_interviewed['A7']['A(Asian or Pacific Islander)'] / table_interviewed['A7']['All'] * 100 # Hispanic
1528 / 44989 * 100
table_interviewed['A15']['W(White)'] / table_interviewed['A15']['All'] * 100 # White
table_interviewed['A15']['B(Black)'] / table_interviewed['A15']['All'] * 100 # Black
table_interviewed['A15']['H(Hispanic)'] / table_interviewed['A15']['All'] * 100 # Hispanic
table_interviewed['A15']['A(Asian or Pacific Islander)'] / table_interviewed['A15']['All'] * 100 # Hispanic
3445 / 18058 * 100
table_interviewed['B2']['W(White)'] / table_interviewed['B2']['All'] * 100 # White
table_interviewed['B2']['B(Black)'] / table_interviewed['B2']['All'] * 100 # Black
table_interviewed['B2']['H(Hispanic)'] / table_interviewed['B2']['All'] * 100 # Hispanic
table_interviewed['B2']['A(Asian or Pacific Islander)'] / table_interviewed['B2']['All'] * 100 # Hispanic
8378 / 68225 * 100
table_interviewed['B3']['W(White)'] / table_interviewed['B3']['All'] * 100 # White
table_interviewed['B3']['B(Black)'] / table_interviewed['B3']['All'] * 100 # Black
table_interviewed['B3']['H(Hispanic)'] / table_interviewed['B3']['All'] * 100 # Hispanic
table_interviewed['B3']['A(Asian or Pacific Islander)'] / table_interviewed['B3']['All'] * 100 # Hispanic
2926 / 24268 * 100
table_interviewed['C6']['W(White)'] / table_interviewed['C6']['All'] * 100 # White
table_interviewed['C6']['B(Black)'] / table_interviewed['C6']['All'] * 100 # Black
table_interviewed['C6']['H(Hispanic)'] / table_interviewed['C6']['All'] * 100 # Hispanic
table_interviewed['C6']['A(Asian or Pacific Islander)'] / table_interviewed['C6']['All'] * 100 # Hispanic
3621 / 35660 * 100
table_interviewed['C11']['W(White)'] / table_interviewed['C11']['All'] * 100 # White
table_interviewed['C11']['B(Black)'] / table_interviewed['C11']['All'] * 100 # Black
table_interviewed['C11']['H(Hispanic)'] / table_interviewed['C11']['All'] * 100 # Hispanic
table_interviewed['C11']['A(Asian or Pacific Islander)'] / table_interviewed['C11']['All'] * 100 # Hispanic
19110 / 124489 * 100
table_interviewed['D4']['W(White)'] / table_interviewed['D4']['All'] * 100 # White
table_interviewed['D4']['B(Black)'] / table_interviewed['D4']['All'] * 100 # Black
table_interviewed['D4']['H(Hispanic)'] / table_interviewed['D4']['All'] * 100 # Hispanic
table_interviewed['D4']['A(Asian or Pacific Islander)'] / table_interviewed['D4']['All'] * 100 # Hispanic
43 / 49787 * 100
table_interviewed['D14']['W(White)'] / table_interviewed['D14']['All'] * 100 # White
table_interviewed['D14']['B(Black)'] / table_interviewed['D14']['All'] * 100 # Black
table_interviewed['D14']['H(Hispanic)'] / table_interviewed['D14']['All'] * 100 # Hispanic
table_interviewed['D14']['A(Asian or Pacific Islander)'] / table_interviewed['D14']['All'] * 100 # Hispanic
1128 / 67529 * 100
table_interviewed['E5']['W(White)'] / table_interviewed['E5']['All'] * 100 # White
table_interviewed['E5']['B(Black)'] / table_interviewed['E5']['All'] * 100 # Black
table_interviewed['E5']['H(Hispanic)'] / table_interviewed['E5']['All'] * 100 # Hispanic
table_interviewed['E5']['A(Asian or Pacific Islander)'] / table_interviewed['E5']['All'] * 100 # Hispanic
761 / 32759 * 100
table_interviewed['E13']['W(White)'] / table_interviewed['E13']['All'] * 100 # White
table_interviewed['E13']['B(Black)'] / table_interviewed['E13']['All'] * 100 # Black
table_interviewed['E13']['H(Hispanic)'] / table_interviewed['E13']['All'] * 100 # Hispanic
table_interviewed['E13']['A(Asian or Pacific Islander)'] / table_interviewed['E13']['All'] * 100 # Hispanic
1965 / 39240 * 100
table_interviewed['E18']['W(White)'] / table_interviewed['E18']['All'] * 100 # White
table_interviewed['E18']['B(Black)'] / table_interviewed['E18']['All'] * 100 # Black
table_interviewed['E18']['H(Hispanic)'] / table_interviewed['E18']['All'] * 100 # Hispanic
table_interviewed['E18']['A(Asian or Pacific Islander)'] / table_interviewed['E18']['All'] * 100 # Hispanic
5087 / 64299 * 100
Image("../input/poverty/2011-2015-Boston-in-Context-Tables_p28.png")
columns = ['DISTRICT','INTERVIEWED_PEOPLE_POPULATION_BASIS','POVERTY']
data = [['A1',1.94,16.1],
        ['A7',3.39,20.3],
        ['A15',19.07,20.3],
        ['B2',3.83,39.8],
        ['B3',12.05,22.1],
        ['C6',10.15,17.6],
        ['C11',15.35,22.9],
        ['D4',0.86,13.2],
        ['D14',1.67,23.8],
        ['E5',2.32,7.3],
        ['E13',5,18.3],
        ['E18',7.91,11.7]
       ]

df = pd.DataFrame(data,columns=columns)

fig, ax = plt.subplots()

df.plot('POVERTY','INTERVIEWED_PEOPLE_POPULATION_BASIS',kind='scatter',ax=ax)

for k, v in df.iterrows():
             ax.annotate(v[0],xy=(v[2],v[1]),size=10)
plt.xlabel('POVERTY RATE')
plt.ylabel('INTERVIEWED PEOPLE BASED ON POPULATION BASIS')
plt.show()