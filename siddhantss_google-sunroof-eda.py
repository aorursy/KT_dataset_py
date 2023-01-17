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
from IPython.display import display

import matplotlib.pyplot as plt

import chart_studio.plotly as py

import plotly.graph_objs as go

import plotly.tools as tls



from tabulate import tabulate

import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics



#graph inline

%matplotlib inline
df_state = pd.read_csv('../input/google-project-sunroof/project-sunroof-state-09082017.csv')

df_city = pd.read_csv('../input/google-project-sunroof/project-sunroof-city-09082017.csv')
def missing_value(df):

    percent_missing = df.isnull().sum() * 100 / len(df)

    q = list(percent_missing)

    missing_value_df = pd.DataFrame({'Column Name': df.columns, 'Percent Missing % ': q})

    return missing_value_df
def state_conversion():

    us_state_abbrev = {

        'Alabama': 'AL',

        'Alaska': 'AK',

        'Arizona': 'AZ',

        'Arkansas': 'AR',

        'California': 'CA',

        'Colorado': 'CO',

        'Connecticut': 'CT',

        'Delaware': 'DE',

        'Florida': 'FL',

        'Georgia': 'GA',

        'Hawaii': 'HI',

        'Idaho': 'ID',

        'Illinois': 'IL',

        'Indiana': 'IN',

        'Iowa': 'IA',

        'Kansas': 'KS',

        'Kentucky': 'KY',

        'Louisiana': 'LA',

        'Maine': 'ME',

        'Maryland': 'MD',

        'Massachusetts': 'MA',

        'Michigan': 'MI',

        'Minnesota': 'MN',

        'Mississippi': 'MS',

        'Missouri': 'MO',

        'Montana': 'MT',

        'Nebraska': 'NE',

        'Nevada': 'NV',

        'New Hampshire': 'NH',

        'New Jersey': 'NJ',

        'New Mexico': 'NM',

        'New York': 'NY',

        'North Carolina': 'NC',

        'North Dakota': 'ND',

        'Ohio': 'OH',

        'Oklahoma': 'OK',

        'Oregon': 'OR',

        'Pennsylvania': 'PA',

        'Rhode Island': 'RI',

        'South Carolina': 'SC',

        'South Dakota': 'SD',

        'Tennessee': 'TN',

        'Texas': 'TX',

        'Utah': 'UT',

        'Vermont': 'VT',

        'Virginia': 'VA',

        'Washington': 'WA',

        'West Virginia': 'WV',

        'Wisconsin': 'WI',

        'Wyoming': 'WY',

    }

    return us_state_abbrev
#to display all the columns of the dataframe

pd.options.display.max_columns = None
df_state.info()
#missing value percentage of state data

missing_value(df_state)
#missing value percentage of city data

missing_value(df_city)
df_state['state_abbr'] = df_state['state_name'].map(state_conversion()) #state conversion function

df_state = df_state[df_state.state_abbr.notna()] #handling na

dup = df_state[df_state.state_name.duplicated()]

df_state.drop_duplicates(subset ="state_name", keep = 'first', inplace = True) #dropping duplicates
df_city['state_abbr'] = df_city['state_name'].map(state_conversion())

df_city = df_city[df_city.state_abbr.notna()]

dup_city = df_city[df_city.duplicated(['region_name','state_name'])]

dup_city
df_city[df_city['percent_qualified'] == 100.0]

citi = df_city.groupby(by = ['state_name'], axis = 0)['percent_qualified'].count()

citi = pd.DataFrame(citi).reset_index()

citi.sort_values(by= ['percent_qualified'], ascending = False, inplace = True)

citi.head()


#choropleth plot



#data to be inputed

data = [go.Choropleth(

    autocolorscale = True,

    locations = df_state.state_abbr,

    z = df_state['yearly_sunlight_kwh_total'],

    locationmode = 'USA-states',

    text = df_state['state_name'],

    marker = go.choropleth.Marker(

        line = go.choropleth.marker.Line(

            color = 'rgb(255,255,255)',

            width = 2

        )),

    colorbar = go.choropleth.ColorBar(

        title = "Total Yearly Sunlight (kwh)")

)]



#layout of the map

layout = go.Layout(

    title = go.layout.Title(

        text = 'Statewise Yearly Sunlight Received (kwh)<br>(Hover for Breakdown)'

    ),

    geo = go.layout.Geo(

        scope = 'usa',

        projection = go.layout.geo.Projection(type = 'albers usa'),

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)'),

)



fig = go.Figure(data = data, layout = layout)

iplot(fig)
df_state.sort_values(by= ['yearly_sunlight_kwh_total', 'state_name'], ascending = False, inplace = True)

plt.figure(figsize=(12,10))

plt.barh(df_state.state_name.head(20),

         df_state.yearly_sunlight_kwh_total.head(20),

         align='center', alpha=0.8)

plt.xlabel('Total Yearly Sunlight-kwh')

plt.ylabel('States')

plt.title('Top 20 states receiving the Maximum Sunlight')

ax = plt.gca()

from matplotlib import ticker

formatter = ticker.ScalarFormatter(useMathText=True)

formatter.set_scientific(True) 

formatter.set_powerlimits((-1,1)) 

ax.xaxis.set_major_formatter(formatter) 

ax.invert_yaxis()

plt.show()

plt.tight_layout()
data = [go.Choropleth(

    autocolorscale = True,

    locations = df_state.state_abbr,

    z = df_state['count_qualified'],

    locationmode = 'USA-states',

    text = df_state['state_name'],

    marker = go.choropleth.Marker(

        line = go.choropleth.marker.Line(

            color = 'rgb(255,255,255)',

            width = 2

        )),

    colorbar = go.choropleth.ColorBar(

        title = "Count of no. of buuilding")

)]



layout = go.Layout(

    title = go.layout.Title(

        text = 'Number of Buildings Qualififed for Solar, Statewise<br> (Hover for Breakdown)'

    ),

    geo = go.layout.Geo(

        scope = 'usa',

        projection = go.layout.geo.Projection(type = 'albers usa'),

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)'),

)



fig = go.Figure(data = data, layout = layout)

iplot(fig)
df_state.sort_values(by= ['count_qualified', 'state_name'], ascending = False, inplace = True)

plt.figure(figsize=(12,8))

plt.barh(df_state.state_name.head(20),

         df_state.count_qualified.head(20), 

         align='center', alpha=0.8)

plt.xlabel('Count Qualified')

plt.title('Top 20 states where the Buildings are suitable for Solar in Google Maps')

ax = plt.gca()



from matplotlib import ticker

formatter = ticker.ScalarFormatter(useMathText=True)

formatter.set_scientific(True) 

formatter.set_powerlimits((-1,1)) 

ax.xaxis.set_major_formatter(formatter) 



for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+0.1, i.get_y()+0.31, \

            str(round((i.get_width()), 2)), color='black')

ax.invert_yaxis()

plt.show()

plt.tight_layout()
plt.figure(figsize=(12,8))

plt.barh(citi.state_name.head(20),

         citi.percent_qualified.head(20),

         align='center', alpha=0.8)

plt.xlabel('Count of hundred percent Qualified Regions')

plt.ylabel('States')

plt.title('States with number of Regions, Hundred percent Qualified for Sunroof')

ax = plt.gca()

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+0.1, i.get_y()+0.31, \

            str(round((i.get_width()), 2)), color='black')

ax.invert_yaxis()

plt.show()

plt.tight_layout()
df_state.sort_values(by= ['number_of_panels_total', 'state_name'], ascending = False, inplace = True)

plt.figure(figsize=(12,10))

plt.barh(df_state.state_name.head(20),

         df_state.number_of_panels_total.head(20),

         align='center', alpha=0.8)

plt.xlabel('Number of Panels')

plt.ylabel('States')

plt.title('Top 20 states with maximum number of Solar Panels Possible')

ax = plt.gca()

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+0.1, i.get_y()+0.31, \

            str(round((i.get_width()), 2)), color='black')

from matplotlib import ticker

formatter = ticker.ScalarFormatter(useMathText=True)

formatter.set_scientific(True) 

formatter.set_powerlimits((-1,1)) 

ax.xaxis.set_major_formatter(formatter) 

ax.invert_yaxis()

plt.show()

plt.tight_layout()
df_city.sort_values(by= ['carbon_offset_metric_tons'], ascending = False, inplace = True)



tt = df_city['region_name'].apply(str)+'<br>'+'Carbon offset(metric tons): ' + df_city['carbon_offset_metric_tons'].apply(str)

#limits = [(0,1000),(1000,10000),(10000,100000),(100000,1000000),(1000000,10000000)]

limits =[(0,10000000)]

colors = ["rgb(255,133,27)"]

#colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)","lightgrey"]

cities = []

scale = 5000



for i in range(len(limits)):

    lim = limits[i]

    df_sub = df_city[lim[0]:lim[1]]

    city = go.Scattergeo(

        locationmode = 'USA-states',

        lon = df_sub['lng_avg'].head(50),

        lat = df_sub['lat_avg'].head(50),

        text = tt,

        marker = go.scattergeo.Marker(

            size = df_sub['carbon_offset_metric_tons']/1e4,

            color = colors[i],

            line = go.scattergeo.marker.Line(

                width=0.5, color='rgb(40,40,40)'

            ),

            sizemode = 'area'

        ),

        name = '{0} - {1}'.format(lim[0],lim[1]) )

    cities.append(city)



layout = go.Layout(

        title = go.layout.Title(

            text = 'Top 50 Cities with Potential carbon Offset<br>(Hover for Break Down)'

        ),

        showlegend = False,

        geo = go.layout.Geo(

            scope = 'usa',

            projection = go.layout.geo.Projection(

                type='albers usa'

            ),

            showland = True,

            landcolor = 'rgb(217, 217, 217)',

            subunitwidth=1,

            countrywidth=1,

            subunitcolor="rgb(255, 255, 255)",

            countrycolor="rgb(255, 255, 255)"

        )

    )



fig = go.Figure(data=cities, layout=layout)

iplot(fig, filename='d3-bubble-map-populations')
df_city = df_city[df_city.carbon_offset_metric_tons.notna()]



#getting the city where Carbon Offset Potential is max

carbon = df_city.groupby(by = ['state_name'], axis = 0)['carbon_offset_metric_tons'].sum() 

carb = pd.DataFrame(carbon).reset_index()

carb.sort_values(by= ['carbon_offset_metric_tons'], ascending = False, inplace = True)

carb.head(10)
plt.figure(figsize=(12,10))

#barplot

plt.barh(carb.state_name.head(20),

         carb.carbon_offset_metric_tons.head(20),

         align='center', alpha=0.8)

plt.xlabel('Carbon Offset (metric tons)')

plt.ylabel('States')

plt.title('Top 20 States with maximum potential Carbon Offset Regions')

ax = plt.gca()

from matplotlib import ticker

formatter = ticker.ScalarFormatter(useMathText=True)

formatter.set_scientific(True) 

formatter.set_powerlimits((-1,1)) 

ax.xaxis.set_major_formatter(formatter) 

ax.invert_yaxis()

plt.show()

plt.tight_layout()
import json

#for texas data

df_texas = df_state[df_state['state_name'] == 'Texas']

df_texas

#loading json data

bucket_texas = df_texas['install_size_kw_buckets_json'].apply(json.loads) 

#converting into list

buckets_texas = list(bucket_texas)

#converting into dataframe

buckpd_texas = pd.DataFrame(buckets_texas).reset_index(drop=True) 

buckpd_texas.dropna(inplace =True)

buckpd_texas_melt = pd.melt(buckpd_texas) #converting into one column



buckpd_texas_melt[['bucket_kW','no_of_buildings']] = pd.DataFrame(buckpd_texas_melt.value.values.tolist(), 

                                                                  index= buckpd_texas_melt.index)

#converting into ranges

bins = pd.cut(buckpd_texas_melt['bucket_kW'], [0, 100, 1000]) 

tx = buckpd_texas_melt.groupby(bins)['bucket_kW'].agg(['count', 'sum'])

tx.columns = ['count_tx','sum_tx']

tx['range'] = ['0 - 100', '100 - 1000']

tx


#for florida

df_florida = df_state[df_state['state_name'] == 'Florida']



bucket_florida = df_florida['install_size_kw_buckets_json'].apply(json.loads)

buckets_florida = list(bucket_florida)

buckpd_florida = pd.DataFrame(buckets_florida).reset_index(drop=True)

buckpd_florida.dropna(inplace =True)

buckpd_florida_melt = pd.melt(buckpd_florida)



buckpd_florida_melt[['bucket_kW','no_of_buildings']] = pd.DataFrame(buckpd_florida_melt.value.values.tolist(), 

                                                                  index= buckpd_florida_melt.index)

bins = pd.cut(buckpd_florida_melt['bucket_kW'], [0, 100, 1000])

fl = buckpd_florida_melt.groupby(bins)['bucket_kW'].agg(['count', 'sum'])

fl.columns = ['count_fl','sum_fl']

fl['range'] = ['0 - 100', '100 - 1000']

fl


#for california

df_cali = df_state[df_state['state_name'] == 'California']



bucket_cali = df_cali['install_size_kw_buckets_json'].apply(json.loads)

buckets_cali = list(bucket_cali)

buckpd_cali = pd.DataFrame(buckets_cali).reset_index(drop=True)

buckpd_cali.dropna(inplace =True)

buckpd_cali_melt = pd.melt(buckpd_cali)



buckpd_cali_melt[['bucket_kW','no_of_buildings']] = pd.DataFrame(buckpd_cali_melt.value.values.tolist(), 

                                                                  index= buckpd_cali_melt.index)

bins = pd.cut(buckpd_cali_melt['bucket_kW'], [0, 100, 1000])

ca = buckpd_cali_melt.groupby(bins)['bucket_kW'].agg(['count', 'sum'])

ca.columns = ['count_ca','sum_ca']

ca['range'] = ['0 - 100', '100 - 1000']

ca
#merging into one

df_txfl = pd.merge(tx,fl)

df_merged = pd.merge(ca,df_txfl)

df_merged


N = 2



ind = np.arange(N) # the x locations for the groups

width = 0.25       # the width of the bars



fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

rects1 = ax.bar(ind, df_merged['sum_tx'], width, color='royalblue')



rects2 = ax.bar(ind+width, df_merged['sum_fl'], width, color='seagreen')



rects3 = ax.bar(ind+width+width, df_merged['sum_ca'], width, color='m')



# add some

ax.set_xlabel('Solar Panel Buckets')

ax.set_ylabel('Number of Buildings')

ax.set_title('Rooftop solar capacity distribution\n(Top 3 States with maximum carbon offset potential regions)')

ax.set_xticks(ind + width +width /10)

ax.set_xticklabels( ['(0 - 100)kW', '(0.1 - 1)mW'] )



ax.legend( (rects1[0], rects2[0], rects3[0]), ('Texas', 'Florida', 'California') )





plt.show()