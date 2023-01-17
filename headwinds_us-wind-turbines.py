# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load
wind = pd.read_csv("../input/uswtdb/uswtdb_v1_0_20180419.csv")   


state_code_to_name = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'GU': 'Guam',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

wind['state'] = wind['t_state'].apply(lambda x: state_code_to_name[x])
# Clean
print("null?: ", wind.isnull().values.any())
print("null count: ", wind.isnull().sum().sum())
# I want to count the number of models and manufacturers
# https://stackoverflow.com/questions/29791785/python-pandas-add-a-column-to-my-dataframe-that-counts-a-variable
df = wind
df['model_count'] = df.groupby('t_model')['t_model'].transform('count')
df['manu_count'] = df.groupby('t_manu')['t_manu'].transform('count')
df_set = df[['state','t_state','p_name','p_year','t_manu','manu_count','t_model','model_count','t_cap','p_cap','t_rd','xlong','ylat']]

wind_set_two_years = df_set.query('p_year>2016')
wind_set_two_years.head()
#Monthly Mean
# I want to roll up the states 
# https://chrisalbon.com/python/data_wrangling/pandas_apply_operations_to_groups/
groupby_state = df['manu_count'].groupby(df['state'])

# groupby_state.mean()
groupby_state.describe()
# Visualize
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = wind_set_two_years['xlong'],
        lat = wind_set_two_years['ylat'],
        text = wind_set_two_years['p_name'] + ' | Rotor Diameter: ' + wind_set_two_years['t_rd'].astype(str),
        mode = 'markers',
        marker = dict(
            size = wind_set_two_years['t_rd'] / 8,
            opacity = 0.9,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = wind_set_two_years['p_cap'],
            cmax = wind_set_two_years['p_cap'].max
            (),
            colorbar=dict(
                title="Power Capacity"
            )
        ))]

layout = dict(
        title = 'US Windturbines 2016 - 2018 <br><span style="font-size:12px">(Circle shows Rotor Size - Hover for windturbine names)</span>',
        colorbar = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='us-windturbines' )

two_years_copy = wind_set_two_years.copy()

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

for col in two_years_copy.columns:
    two_years_copy[col] = two_years_copy[col].astype(str)

two_years_copy['text'] = two_years_copy['state'] + '<br>' + two_years_copy['model_count']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = two_years_copy['t_state'],
        z = two_years_copy['model_count'],
        locationmode = 'USA-states',
        text = two_years_copy['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Model Count")
        ) ]

layout = dict(
        title = '2016 - 2018 US Wind Turbine Purchases<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )
agg_wind_two_years = wind_set_two_years['model_count'].groupby(wind_set_two_years['state']).agg('count')
# agg_wind_2015
df_wind_two_years = pd.DataFrame({'state':agg_wind_two_years.index, 'model_count':agg_wind_two_years.values}).sort_values('model_count')

data = [dict(
  type = 'bar',
  x = df_wind_two_years.state,
  y = df_wind_two_years.model_count,
)]

layout = dict(
  title = '<b>US Wind Turbine Distribution</b><br><span style="font-size: 12px">Total Models Built By State between 2016 and 2018</span>',
  xaxis = dict(title = 'States'),
  yaxis = dict(title = 'Model Count', range = [0,1500]),
  updatemenus = [dict(
        x = 0.85,
        y = 1.15,
        xref = 'paper',
        yref = 'paper',
        yanchor = 'top',
        active = 1,
        showactive = False
  )]
)

iplot({'data': data,'layout': layout}, validate=False)
# who are the top manufacturers between 2016-2018? How do I sum up unique?! So I dont get 2 columns...
# top_ten_manufacturers = wind_set_two_years.sort_values('manu_count').head(10)
top_manufacturers = wind_set_two_years.sort_values('manu_count')

data = [dict(
  type = 'bar',
  x = top_manufacturers.t_manu,
  y = top_manufacturers.manu_count,
)]

layout = dict(
  title = '<b>Top Wind Turbine Manufacturers between 2016 - 2018</b>',
  xaxis = dict(title = 'Manufacturer'),
  yaxis = dict(title = 'Wind Turbines Sold', range = [0,25000]),
  updatemenus = [dict(
        x = 0.85,
        y = 1.15,
        xref = 'paper',
        yref = 'paper',
        yanchor = 'top',
        active = 1,
        showactive = False
  )]
)

iplot({'data': data,'layout': layout}, validate=False)


wind_set_2015 = df_set.query('p_year==2015')

# wind_set_2015.shape (4287, 12)
'''
Name	Average retail price (cents/kWh)	Net summer capacity (MW)	Net generation (MWh)	Total retail sales (MWh)
'''
consumed = pd.read_csv("../input/us-states-energy-profiles-2015/us_state_energy_2015.csv")

# join the Net_summer_capacity and find out what percentage is wind turbine based on the t_cap column
# joined = pd.concat([df_a, consumed], axis=1)
agg_wind_2015 = wind_set_2015['t_cap'].groupby(wind_set_2015['state']).agg('sum')
# agg_wind_2015
df_wind_2015 = pd.DataFrame({'state':agg_wind_2015.index, 't_cap':agg_wind_2015.values}).sort_values('state')

consumed.columns = ['state', 'Average_retail_price', 'Net_summer_capacity','Net_generation','Total_retail_sales']
consumed = consumed.sort_values('state')

merged = pd.merge(df_wind_2015, consumed, on='state', how='left')
merged
trace1 = go.Bar(
    type = 'bar',
  x = merged.state,
  y = merged.t_cap,
    name='Wind Turbine'
)
trace2 = go.Bar(
    type = 'bar',
  x = merged.state,
  y = merged.Net_generation,
    name='All Energy Sources'
)

merged_data = [trace1, trace2]

merged_layout = dict(
  title = '<b>Net Capacity vs Wind Turbine Capacity</b><br>US States, 2015',
  xaxis = dict(title = 'States'),
  yaxis = dict(title = 'Capacity', range = [0,500000000]),
  updatemenus = [dict(
        x = 0.85,
        y = 1.15,
        xref = 'paper',
        yref = 'paper',
        yanchor = 'top',
        active = 1,
        showactive = False
  )]
)

iplot({'data': merged_data,'layout': merged_layout}, validate=False)
