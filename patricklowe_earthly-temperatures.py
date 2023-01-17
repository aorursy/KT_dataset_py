import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

!pip install country_converter

import country_converter as coco



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')

df['Country'] = df['Country'].str.replace(' (Europe)','')
# Extract dates

df['day'] = df['dt'].str[-2:]

df['year'] = df['dt'].str[:4]

df['month'] = df['dt'].str[5:7] 
yearly_max = df.groupby(by=["Country",'year'])['AverageTemperature'].max().reset_index()

yearly_min = df.groupby(by=["Country",'year'])['AverageTemperature'].min().reset_index()
yearly_diff = yearly_max.drop('AverageTemperature',1)

yearly_diff['change'] = yearly_max['AverageTemperature'] - yearly_min['AverageTemperature']

yearly_diff.head(3)
long_code = pd.read_csv('../input/long-country-codes/country_codes.csv')

short_code = pd.read_csv('../input/country-codes/country_codes.csv')

short_code.head(3)
yearly_diff['2code'] = yearly_diff['Country'].map(long_code.set_index('Name')['Code'])

yearly_diff['3code'] = yearly_diff['2code'].map(short_code.set_index('Alpha-2 code')['Alpha-3 code'])

yearly_diff = yearly_diff.sort_values(by=['year','3code'])

yearly_diff['year'] = yearly_diff['year'].astype(int)
fig = px.choropleth(yearly_diff[ yearly_diff['year'] > 1899], 

                    locations="3code",

                    color_continuous_scale="YlOrBr",

                    color="change",

                    animation_frame="year",

                    )



fig.update_layout(

    title_text = 'Yearly Max - Min',

    template="plotly_dark",

)



fig.update_geos(

    projection_type="miller",

    #fitbounds="locations",

    resolution=50,

    landcolor="silver",

    oceancolor="cadetblue",

    showocean=True,

    showlakes=False,

)



fig.show()
yearly_avg = df.groupby(by=["Country",'year'])['AverageTemperature'].mean().reset_index()

yearly_avg['year'] = yearly_avg['year'].astype(int)

yearly_avg['2code'] = yearly_avg['Country'].map(long_code.set_index('Name')['Code'])

yearly_avg['3code'] = yearly_avg['2code'].map(short_code.set_index('Alpha-2 code')['Alpha-3 code'])

yearly_avg = yearly_avg.sort_values(by=['year','3code'])

yearly_avg['year'] = yearly_avg['year'].astype(int)

yearly_avg = yearly_avg[ yearly_avg['year'] > 1949]

yearly_avg.head(3)
yearly_avg2 = yearly_avg.groupby(by=['year'])['AverageTemperature'].mean().reset_index()

yearly_avg2.head()
"""

fig = make_subplots(

    rows=1, 

    cols=2,

    subplot_titles=("Average Temp", "Global Average"),

    specs=[ [{"type": "scattergeo"},{"type": "bar"}]]

)



for step in yearly_avg["year"].unique():

    fig.add_trace(

        go.Choropleth(

                    visible=False,

                    colorbar_title = "Temp (C)", 

                    locations = yearly_avg["3code"],

                    colorscale='thermal',

                    z = yearly_avg["AverageTemperature"],

                    ),

        row=1,

        col=1

    )

    

    fig.append_trace(

        go.Scatter(

            x = yearly_avg2['year'], 

            y = yearly_avg2['AverageTemperature'],

            mode='lines',

            name='lines',

            showlegend=False,),

        row=1, 

        col=2

    )

    

 



fig.data[0].visible = True



years = yearly_avg['year'].unique()

steps = []

j = 0

for i in range(0, len(fig.data), 2):

    step = dict(method="restyle", args=["visible", [False] * len(fig.data)],label ='Year:{}'.format(years[j]))

    j += 1

    step['args'][1][i] = True

    step['args'][1][i+1] = True

    steps.append(step)



sliders = [dict(active=0,pad={"t": 50},steps=steps)]



# Update geo subplot properties

fig.update_geos(

    projection_type="miller",

    fitbounds="locations",

    resolution=50,

    landcolor="silver",

    oceancolor="cadetblue",

    showocean=True,

    showlakes=False,

)



# Set theme, margin, and annotation in layout

fig.update_layout(

    template="plotly_dark",

    height = 800,

    yaxis=dict(range=[18.2, 19.8]),

    margin=dict(r=10, t=55, b=40, l=20),

    title_text = 'Global Average Temp (C)',

    sliders=sliders,

    legend=dict(

        orientation="h",

        yanchor="bottom",

        y=1.02,

        xanchor="right",

        x=1

    ),

)



fig.show()

"""
df['year'] = df['year'].astype(int)

uncert = df[ df['year'] > 1949]

uncert = uncert[ uncert['Country'] == 'Ireland']

uncert.head(3)
fig = go.Figure([

    go.Scatter(

        name='Temp',

        x=uncert['dt'],

        y=uncert['AverageTemperature'],

        mode='lines',

        line=dict(color='rgb(75, 75, 75)'),

    ),

    go.Scatter(

        name='Upper Bound',

        x=uncert['dt'],

        y=uncert['AverageTemperature'] + uncert['AverageTemperatureUncertainty'],

        mode='lines',

        marker=dict(color="#444"),

        line=dict(width=0),

        showlegend=False

    ),



    go.Scatter(

        name='Lower Bound',

        x=uncert['dt'],

        y=uncert['AverageTemperature'] - uncert['AverageTemperatureUncertainty'],

        marker=dict(color="#444"),

        line=dict(width=0),

        mode='lines',

        fillcolor='rgba(255, 0, 0, 0.7)',

        fill='tonexty',

        showlegend=False

    )

])

fig.update_layout(

    yaxis_title='Temp(C)',

    title='Temp with Uncertainty Range (C)',

    hovermode="x",

    template="plotly_dark",

    height = 800,

)

fig.show()
hemi = df

hemi['2code'] = df['Country'].map(long_code.set_index('Name')['Code'])

hemi['3code'] = hemi['2code'].map(short_code.set_index('Alpha-2 code')['Alpha-3 code'])

hemi = hemi.sort_values(by=['year','3code'])

hemi = hemi.drop(['dt','day','2code'],1)

hemi['year'] = hemi['year'].astype(int)

hemi['hemisphere'] = hemi['Country'].isin(['Indonesia','Brazil','Ecuador','New Caledonia','Fiji','Falkland Islands','Swaziland','Peru',

                                                   'Chile','Bolivia','Paraguay','Argentina','Uruguay','Angola','Namibia','Botswana','South Africa',

                                                   'Zimbabwe','Tanzania','Mozambique','Zambia','Lesotho','Madagascar','Australia','Papua New Guinea',

                                                   'New Zealand','Solomon Islands','Antarctica']).astype(int)

#hemi['hemisphere'] = hemi['hemisphere'].replace(False,'Northern')

#hemi['hemisphere'] = hemi['hemisphere'].replace(True,'Southern')

hemi
# Average for each year by hemi

hemi_avg = hemi.groupby(by=["hemisphere",'year'])['AverageTemperature'].mean().reset_index()

hemi_avg2 = hemi.groupby(by=["hemisphere",'year'])['AverageTemperatureUncertainty'].mean().reset_index()

hemi_avg['Uncert'] = hemi_avg2['AverageTemperatureUncertainty']

hemi_avg = hemi_avg[hemi_avg['year'] > 1859]

hemi_avg.head(3)
hemi_country = hemi.drop(['AverageTemperatureUncertainty','AverageTemperature','Country','year','month'],1).drop_duplicates() 
colorscale = ['#06C','#F66']



fig = make_subplots(

    rows=1, 

    cols=2,

    subplot_titles=("Country by Hemisphere", "Hemisphere Average"),

    specs=[ [{"type": "scattergeo"},{"type": "bar"}]]

)



# GeoMap

fig.add_trace(

    go.Choropleth(

        locations = hemi_country['3code'],

        z = hemi_country['hemisphere'],        

        colorscale = colorscale,

        showlegend=False,

        showscale=False,

    ),

    row=1, col=1

)



# Southern Hemisphere

fig.add_trace(

    go.Scatter(

        x = hemi_avg[hemi_avg['hemisphere'] == 1]['year'], 

        y = hemi_avg[hemi_avg['hemisphere'] == 1]['AverageTemperature'],

        name='Southern',

        marker_color='#F66',

        ),

    row=1, col=2

)



# Northern Hemisphere

fig.add_trace(

    go.Scatter(

        x = hemi_avg[hemi_avg['hemisphere'] == 0]['year'], 

        y = hemi_avg[hemi_avg['hemisphere'] == 0]['AverageTemperature'],

        name='Northern',

        marker_color='#06C',

        ),

    row=1, col=2

)



# Uncertainty for Northern Hemi, Lowerbound

fig.add_trace(

    go.Scatter(

        name='N.Hemi, Lower',

        x = hemi_avg[hemi_avg['hemisphere'] == 0]['year'],

        y = hemi_avg[hemi_avg['hemisphere'] == 0]['AverageTemperature'] - hemi_avg[hemi_avg['hemisphere'] == 0]['Uncert'],

        marker=dict(color="#444"),

        line=dict(width=0),

        mode='lines',

        fillcolor='rgba(102,255,255,0.5)',

        showlegend=False),

    row=1, col=2

)



# Uncertainty for Northern Hemi, Upperbound

fig.add_trace(

    go.Scatter(

        name='N.Hemi, Upper',

        x = hemi_avg[hemi_avg['hemisphere'] == 0]['year'],

        y = hemi_avg[hemi_avg['hemisphere'] == 0]['AverageTemperature'] + hemi_avg[hemi_avg['hemisphere'] == 0]['Uncert'],

        marker=dict(color="#444"),

        line=dict(width=0),

        mode='lines',

        fillcolor='rgba(102,255,255,0.5)',

        fill='tonexty',

        showlegend=False),

    row=1, col=2

)



# Uncertainty for Southern Hemi, Lowerbound

fig.add_trace(

    go.Scatter(

        name='S.Hemi, Lower Bound',

        x = hemi_avg[hemi_avg['hemisphere'] == 1]['year'],

        y = hemi_avg[hemi_avg['hemisphere'] == 1]['AverageTemperature'] - hemi_avg[hemi_avg['hemisphere'] == 1]['Uncert'],

        marker=dict(color="#444"),

        line=dict(width=0),

        mode='lines',

        fillcolor='rgba(255,255,204,0.5)',

        showlegend=False),

    row=1, col=2

)



# Uncertainty for Southern Hemi, Upperbound

fig.add_trace(

    go.Scatter(

        name='S.Hemi, Upper',

        x = hemi_avg[hemi_avg['hemisphere'] == 1]['year'],

        y = hemi_avg[hemi_avg['hemisphere'] == 1]['AverageTemperature'] + hemi_avg[hemi_avg['hemisphere'] == 1]['Uncert'],

        marker=dict(color="#444"),

        line=dict(width=0),

        mode='lines',

        fillcolor='rgba(255,255,204,0.5)',

        fill='tonexty',

        showlegend=False),

    row=1, col=2

)



fig.update_geos(

    projection_type="miller",

    resolution=50,

    landcolor="white",

    oceancolor="cadetblue",

    showocean=True,

    showlakes=False,

)



fig.update_layout(

    template="plotly_dark",

    title_text = 'Global Average Temp (C)',

)



fig.show()
gas = pd.read_csv('../input/grenhouse-gas/gas_emissions(kt_c01_equiv).csv')

gas['temp'] = 0

gas = gas.melt(id_vars=["Country Name", "Country Code"], value_vars=['1970','1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991','1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012'],var_name='year', value_name='CO2')
co2_avg = gas.groupby(by=['Country Code'])['CO2'].mean().reset_index()

co2_avg['CO2'].min()
fig = make_subplots(rows=1,cols=1,

    specs=[ [{"type": "scattergeo"}]]

)



fig.add_trace(

    go.Choropleth(

        locations = co2_avg['Country Code'],

        z = co2_avg['CO2'],        

        zmax=1000000,

        zmin=8000,

    ),

    row=1, col=1

)



fig.update_geos(

    projection_type="miller",

    resolution=50,

    fitbounds='locations',

    landcolor="white",

    oceancolor="cadetblue",

    showocean=True,

    showlakes=False,

)



fig.update_layout(

    template="plotly_dark",

    height=800,

    title_text = '40yr Avg. Global CO2 Emissions (Kt)',

)
# Comparison between Continents

lists = []

countries = gas['Country Name']

for country in countries:

    lists.append( coco.convert(names = country, to = 'continent', not_found=None) )

gas['continents'] = lists

continents = ['America','Asia','Africa','Europe','Oceania']

emissions = gas[gas['continents'].isin(continents) ]

global_emissions = emissions.groupby(by=['continents','year'])['CO2'].mean().reset_index()
cont_temps = hemi

#cont_temps = cont_temps.drop(['AverageTemperatureUncertainty','month','hemisphere'],1)

cont_temps = cont_temps[ cont_temps['year'] > 1969]

cont_temps = cont_temps.groupby(by=['Country','year'])['AverageTemperature'].mean().reset_index()

cont_temps
lists = []

countries = cont_temps['Country']



for country in countries:

    lists.append( coco.convert(names = country, to = 'continent', not_found=None) )

cont_temps['continents'] = lists

continents = ['America','Asia','Africa','Europe','Oceania']

cont_temps = cont_temps[cont_temps['continents'].isin(continents) ]

cont_temps = cont_temps.groupby(by=['continents','year'])['AverageTemperature'].mean().reset_index()
fig = make_subplots(

    rows=2, 

    cols=2,

    subplot_titles=("40yr Avg CO2 Emissions (kt)", "CO2 Emissions by Continent", "Temp"),

    specs=[ [{"type": "scattergeo", "rowspan": 2},{"type": "bar"}],

            [None,{"type": "bar"}] ])



fig.add_trace(

    go.Choropleth(

        locations = co2_avg['Country Code'],

        z = co2_avg['CO2'],        

        zmax=1000000,

        zmin=8000,

        showscale=False,

    ),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(

        x = global_emissions[global_emissions['continents'] == 'Europe']['year'], 

        y = global_emissions[global_emissions['continents'] == 'Europe']['CO2'],

        name='Europe',

        marker_color='#06C',

        ),

    row=1, col=2

)



fig.add_trace(

    go.Scatter(

        x = global_emissions[global_emissions['continents'] == 'Asia']['year'], 

        y = global_emissions[global_emissions['continents'] == 'Asia']['CO2'],

        name='Asia',

        marker_color='#C00',

        ),

    row=1, col=2

)



fig.add_trace(

    go.Scatter(

        x = global_emissions[global_emissions['continents'] == 'America']['year'], 

        y = global_emissions[global_emissions['continents'] == 'America']['CO2'],

        name='America',

        marker_color='#CC0',

        ),

    row=1, col=2

)



fig.add_trace(

    go.Scatter(

        x = global_emissions[global_emissions['continents'] == 'Oceania']['year'], 

        y = global_emissions[global_emissions['continents'] == 'Oceania']['CO2'],

        name='Oceania',

        marker_color='#C9F',

        ),

    row=1, col=2

)



#TEMPERATURE GRAPH



fig.add_trace(

    go.Scatter(

        x = cont_temps[cont_temps['continents'] == 'Oceania']['year'], 

        y = cont_temps[cont_temps['continents'] == 'Oceania']['AverageTemperature'],

        name='Oceania',

        marker_color='#C9F',

        showlegend=False,

        ),

    row=2, col=2

)



fig.add_trace(

    go.Scatter(

        x = cont_temps[cont_temps['continents'] == 'America']['year'], 

        y = cont_temps[cont_temps['continents'] == 'America']['AverageTemperature'],

        name='America',

        marker_color='#CC0',

        showlegend=False,

        ),

    row=2, col=2

)



fig.add_trace(

    go.Scatter(

        x = cont_temps[cont_temps['continents'] == 'Europe']['year'], 

        y = cont_temps[cont_temps['continents'] == 'Europe']['AverageTemperature'],

        name='Euorpe',

        marker_color='#06C',

        showlegend=False,

        ),

    row=2, col=2

)



fig.add_trace(

    go.Scatter(

        x = cont_temps[cont_temps['continents'] == 'Asia']['year'], 

        y = cont_temps[cont_temps['continents'] == 'Asia']['AverageTemperature'],

        name='Asia',

        marker_color='#C00',

        showlegend=False,

        ),

    row=2, col=2

)



fig.update_geos(

    projection_type="miller",

    resolution=50,

    landcolor="white",

    oceancolor="cadetblue",

    showocean=True,

    showlakes=False,

)



fig.update_layout(

    height=800,

    template="plotly_dark",

    title_text = 'Global Average Temp (C)',

)



fig.show()