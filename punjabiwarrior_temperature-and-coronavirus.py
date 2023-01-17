import numpy as np 
import pandas as pd 
from matplotlib import pyplot
import seaborn as sns; sns.set()

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import missingno as msno



%matplotlib inline
mpl.style.use( 'ggplot' )
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)
!pip install missingno
df_total = pd.read_csv("../input/coronavirus/time_series_covid19_confirmed_global.csv")
df_total.head()
countries=['US', 'Canada', 'Germany']
y=df_total.loc[df_total['Country/Region']=='Italy'].iloc[0,4:]
s = pd.DataFrame({'Italy':y})
for c in countries:    
    #pyplot.plot(range(y.shape[0]),y,'r--')
    s[c] = df_total.loc[df_total['Country/Region']==c].iloc[0,4:]
#pyplot.plot(range(y.shape[0]),y,'g-')
pyplot.plot(range(y.shape[0]), s)
plt.gca().legend(('Italy', 'US', 'Canada', 'Germany'))
for r in df_total['Country/Region'].unique():
    if r != 'China':
        pyplot.plot(range(len(df_total.columns)-4), df_total.loc[df_total['Country/Region']==r].iloc[0,4:],label=r)
        #pyplot.legend()
df_recovered=  pd.read_csv("../input/coronavirus/time_series_covid19_recovered_global.csv")
df_recovered.head()
df_deaths=  pd.read_csv("../input/coronavirus/time_series_covid19_deaths_global.csv")
df_deaths.head()
df_active = pd.DataFrame()
df_active1= df_total.loc[:,'1/22/20': ].subtract(df_deaths.loc[:,'1/22/20': ], axis = 1) 
df_active = df_active1.loc[:,'1/22/20': ].subtract(df_recovered.loc[:,'1/22/20': ], axis = 1)
df_active.head()
df_active.insert(0, 'Province/State', df_total['Province/State'])
df_active.insert(1, 'Country/Region', df_total['Country/Region'])
df_active.insert(2, 'Lat', df_total['Lat'])
df_active.insert(3, 'Long', df_total['Long'])
df_active.head() #Calculating active cases, as we have numbers of confirmed, recovered and death
df_cleaned =  pd.DataFrame()
df_cleaned.insert(0, 'Province/State', df_total['Province/State'])
df_cleaned.insert(1, 'Country/Region', df_total['Country/Region'])
df_cleaned.insert(2, 'Lat', df_total['Lat'])
df_cleaned.insert(3, 'Long', df_total['Long'])
df_cleaned.insert(4, 'Confirmed', df_total['4/10/20'])
df_cleaned.insert(5, 'Active', df_active['4/10/20'])
df_cleaned.insert(6, 'Recovered', df_recovered['4/10/20'])
df_cleaned.insert(7, 'Deaths', df_deaths['4/10/20'])
df_cleaned.head() ##A clearer version of cases sorted by province
df = df_cleaned.groupby(['Country/Region']).sum().reset_index()
#df = df.reset_index(drop= True)
df   ##With different countries
df_final = df.sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
df_final.head(10) ##Top 10 Countries with highest number of cases. Worked on this small dataset
df_final.style.background_gradient(cmap='viridis', low=.5, high=0)
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "4pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]
cmap = cmap=sns.diverging_palette(10, 250, as_cmap=True)
bigdf = df_final

bigdf.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '1pt'})\
    .set_caption("Hover to magnify")\
    .set_precision(2)\
    .set_table_styles(magnify())
df_final.head(10)
import plotly as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objects as go
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)  
!pip install plotly
res = [*(range(len(df_total.columns)-4))] 
total_sum_column = pd.DataFrame(columns=['Confirmed', 'Recovered'])
total_sum_column['Confirmed'] =  df_total.sum(axis=0)
total_sum_column['Recovered'] =  df_recovered.sum(axis=0)
total_sum_column = total_sum_column[2:]
total_sum_column
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(
                x = res,
                y = total_sum_column['Confirmed'],
                name="Confirmed"))

fig.add_trace(go.Scatter(
                x = res,
                y=total_sum_column['Recovered'],
                name="Recovered"))
                

fig.update_layout(title_text='Time Series with Rangeslider varying with number of days starting from 22 Jan,2020',
                  xaxis_rangeslider_visible=True)
py.offline.iplot(fig)
import plotly.offline as py
Countries = np.unique(df_final['Country/Region'])
cases = []
for country in Countries:
    cases.append(df_final[df_final['Country/Region'] == country]['Confirmed'].sum())
data = [ dict(
        type = 'choropleth',
        locations = Countries,
        z = cases,
        locationmode = 'country names',
        text = Countries,
        marker = dict(
            line = dict(color = 'rgb(0,0,0)', width = 1)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'World map')
            )
       ]
layout = dict(
    title = 'COVID-19 Confirmed Cases with countries',
    geo = dict(
        showframe = False,
        showocean = True,
        oceancolor = 'rgb(3, 186, 252)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap')
df1=df_final[['Country/Region','Confirmed']]
df1
Temp = pd.read_csv('../input/temperature/GlobalLandTemperaturesByCountry.csv')##Average land surface temperatures from wikipedia
Temp.head()
Temp_clean = Temp[~Temp['Country'].isin(
    ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',
     'United Kingdom', 'Africa', 'South America'])]

Temp_clean = Temp_clean.replace(
   ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],
   ['Denmark', 'France', 'Netherlands', 'United Kingdom'])
countries = np.unique(Temp_clean['Country'])
Avg_temp = []
for country in countries:
    Avg_temp.append(Temp_clean[Temp_clean['Country'] == country]['AverageTemperature'].mean())
data = [ dict(
        type = 'choropleth',
        locations = countries,
        z = Avg_temp,
        locationmode = 'country names',
        text = countries,
        marker = dict(
            line = dict(color = 'rgb(0,0,0)', width = 1)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Average Temp')
            )
       ]

layout = dict(
    title = 'Average Land Temperatures of Different Countries',
    geo = dict(
        showframe = False,
        showocean = True,
        oceancolor = 'rgb(3, 198, 252)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap')
#List of average temperatures in April, to be included as new column 
country = ['US', 'Spian', 'Italy','France', 'Germany', 'China','United Kingdom','Iran', 'Turkey', 'Belgium']
degree = [12.1, 12.9, 12.2, 11.4 , 10,  14.8, 8.7, 17,11.1, 9.8]
Visual = df_final.head(10)
Visual['Avg Temp'] = degree
Visual
sns.jointplot(x="Avg Temp", y="Confirmed", data=Visual, size=4.5)
sns.FacetGrid(Visual, hue="Country/Region", size=5) \
   .map(plt.scatter, "Avg Temp", "Confirmed") \
   .add_legend()















