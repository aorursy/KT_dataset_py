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
import pandas as pd
data="../input/countries-of-the-world/countries of the world.csv"
data_env="../input/environmental-variables-for-world-countries/World_countries_env_vars.csv"
#educ="../input/education-statistics/data.csv"
#gapminder="../input/gapminder/gapminder.tsv"

data=pd.read_csv(data)
data_env=pd.read_csv(data_env)
#educ=pd.read_csv(educ)
#gapminder=pd.read_csv(gapminder, sep='\t')

data_env['Country']=data_env['Country'].astype(str).str.strip()
data['Country']=data['Country'].astype(str).str.strip()
data['Region']=data['Region'].astype(str).str.strip()
data_all=pd.merge(data, data_env, how='inner', on='Country')
data_all=data_all.sort_values('Population', ascending=False)

labels={ 
'Pop. Density (per sq. mi.)': 'Pop. Density (per sq. mi.)', 
'Coastline (coast/area ratio)':'Coastline (coast/area ratio)', 
'Infant mortality (per 1000 births)':'Infant mortality (per 1000 births)', 
'Literacy (%)':'Literacy (%)', 
'Phones (per 1000)':'Phones (per 1000)', 
'Net migration':'Net migration', 
'Population':'Population', 
'Birthrate':'Birthrate' , 
'Deathrate':'Deathrate', 
'GDP ($ per capita)' : 'Gross domestic product per capita', 
'accessibility_to_cities' : 'travel time to cities (minutes)', 
'elevation': 'elevation above sea level (m)', 
'aspect': 'orientation of slope', 
'slope': 'angle of slope', 
'cropland_cover': 'percentage of country covered by cropland (%)', 
'tree_canopy_cover': 'percentage cover by trees > 5m in height (%)', 
'isothermality': 'Isothermality (diurnal range / annual range)', 
'rain_coldestQuart': 'Precipitation of coldest quarter (mm)', 
'rain_driestMonth': 'Precipitation of driest month (mm)', 
'rain_driestQuart': 'Precipitation of driest quarter (mm)', 
'rain_mean_annual': 'Annual precipitation (mm)', 
'rain_seasonailty': 'Precipitation seasonality (coefficient of variation)', 
'rain_warmestQuart': 'Precipitation of warmest quarter (mm)', 
'rain_wettestMonth': 'Precipitation of wettest month (mm)', 
'rain_wettestQuart': 'Precipitation of wettest quarter (mm)', 
'temp_annual_range': 'Temperature annual range (bio05-bio06) (%)', 
'temp_coldestQuart': 'Mean temperature of coldest quarter (degC)', 
'temp_diurnal_range': 'Mean diurnal range (mean of monthly (max temp - min temp)) (degC)', 
'temp_driestQuart': 'Mean temperature of driest quarter (degC)', 
'temp_max_warmestMonth': 'Max temperature of warmest month (degC)', 
'temp_mean_annual': 'Temperature mean annual (degC)', 
'temp_min_coldestMonth': 'Min temperature of coldest month (degC)', 
'temp_seasonality': 'Temperature seasonality (Standard deviation * 100) (degC)', 
'temp_warmestQuart': 'Mean temperature of warmest quarter (degC)', 
'temp_wettestQuart': 'Mean temperature of wettest quarter (degC)', 
'wind': 'Mean wind speed (m/s)', 
'cloudiness': 'Average cloudy days per year (days)'}

data_all['Birthrate']=data_all['Birthrate'].str.replace(',', '.').astype(float)
data_all['Deathrate']=data_all['Deathrate'].str.replace(',', '.').astype(float)
data_all['Net migration']=data_all['Net migration'].str.replace(',', '.').astype(float)
data_all['Pop. Density (per sq. mi.)']=data_all['Pop. Density (per sq. mi.)'].str.replace(',', '.').astype(float)
data_all['Coastline (coast/area ratio)']=data_all['Coastline (coast/area ratio)'].str.replace(',', '.').astype(float)
data_all['Infant mortality (per 1000 births)']=data_all['Infant mortality (per 1000 births)'].str.replace(',', '.').astype(float)
data_all['Literacy (%)']=data_all['Literacy (%)'].str.replace(',', '.').astype(float)
data_all['Phones (per 1000)']=data_all['Phones (per 1000)'].str.replace(',', '.').astype(float)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
import numpy as np

init_notebook_mode(connected=True)


#TO GET A GRAPH IN OTHER AXES JUST CHANGE THE VARIABLE NAMES HERE


#horizontal='Birthrate'
#vertical='Deathrate'

#horizontal='temp_mean_annual'
#vertical='GDP ($ per capita)'

#horizontal='temp_mean_annual'
#vertical='rain_mean_annual'

#horizontal='temp_mean_annual'
#vertical='cloudiness'

horizontal='GDP ($ per capita)'
vertical='Phones (per 1000)'

div=10

trace0 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'ASIA (EX. NEAR EAST)'],
    y=data_all[vertical][data_all['Region'] == 'ASIA (EX. NEAR EAST)'],
    mode='markers',
    name='ASIA',
    text=data_all['Country'][data_all['Region'] == 'ASIA (EX. NEAR EAST)'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'ASIA (EX. NEAR EAST)'])/div)
    )
)



trace1 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'LATIN AMER. & CARIB'],
    y=data_all[vertical][data_all['Region'] == 'LATIN AMER. & CARIB'],
    mode='markers',
    name='LATIN AMER',
    text=data_all['Country'][data_all['Region'] == 'LATIN AMER. & CARIB'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'LATIN AMER. & CARIB'])/div)
    )
)

trace2 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'SUB-SAHARAN AFRICA'],
    y=data_all[vertical][data_all['Region'] == 'SUB-SAHARAN AFRICA'],
    mode='markers',
    name='AFRICA',
    text=data_all['Country'][data_all['Region'] == 'SUB-SAHARAN AFRICA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'SUB-SAHARAN AFRICA'])/div)
    )
)


trace3 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'C.W. OF IND. STATES'],
    y=data_all[vertical][data_all['Region'] == 'C.W. OF IND. STATES'],
    mode='markers',
    name='Former USSR',
    text=data_all['Country'][data_all['Region'] == 'C.W. OF IND. STATES'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'C.W. OF IND. STATES'])/div)
    )
)


trace4 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'WESTERN EUROPE'],
    y=data_all[vertical][data_all['Region'] == 'WESTERN EUROPE'],
    mode='markers',
    name='WESTERN EUROPE',
    text=data_all['Country'][data_all['Region'] == 'WESTERN EUROPE'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'WESTERN EUROPE'])/div)
    )
)

trace5 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'OCEANIA'],
    y=data_all[vertical][data_all['Region'] == 'OCEANIA'],
    mode='markers',
    name='OCEANIA',
    text=data_all['Country'][data_all['Region'] == 'OCEANIA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'OCEANIA'])/div)
    )
)


trace6 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'NEAR EAST'],
    y=data_all[vertical][data_all['Region'] == 'NEAR EAST'],
    mode='markers',
    name='NEAR EAST',
    text=data_all['Country'][data_all['Region'] == 'NEAR EAST'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'NEAR EAST'])/div)
    )
)


trace7 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'EASTERN EUROPE'],
    y=data_all[vertical][data_all['Region'] == 'EASTERN EUROPE'],
    mode='markers',
    name='EASTERN EUROPE',
    text=data_all['Country'][data_all['Region'] == 'EASTERN EUROPE'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'EASTERN EUROPE'])/div)
    )
)


trace8 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'NORTHERN AFRICA'],
    y=data_all[vertical][data_all['Region'] == 'NORTHERN AFRICA'],
    mode='markers',
    name='NORTHERN AFRICA',
    text=data_all['Country'][data_all['Region'] == 'NORTHERN AFRICA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'NORTHERN AFRICA'])/div)
    )
)


trace9 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'NORTHERN AMERICA'],
    y=data_all[vertical][data_all['Region'] == 'NORTHERN AMERICA'],
    mode='markers',
    name='NORTHERN AMERICA',
    text=data_all['Country'][data_all['Region'] == 'NORTHERN AMERICA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'NORTHERN AMERICA'])/div)
    )
)


trace10 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'BALTICS'],
    y=data_all[vertical][data_all['Region'] == 'BALTICS'],
    mode='markers',
    name='BALTICS',
    text=data_all['Country'][data_all['Region'] == 'BALTICS'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'BALTICS'])/div)
    )
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]

layout = go.Layout(
    title=labels[horizontal]+' vs. '+labels[vertical],
    xaxis=dict(
        title=labels[horizontal],
        titlefont=dict(
            family='Courier New, monospace',
            size=18
          
        )
    ),
    yaxis=dict(
        title=labels[vertical],
        titlefont=dict(
            family='Courier New, monospace',
            size=18
            
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
import numpy as np

init_notebook_mode(connected=True)



horizontal='temp_mean_annual'
vertical='cloudiness'  

#horizontal='Birthrate'
#vertical='Deathrate'

#horizontal='temp_mean_annual'
#vertical='GDP ($ per capita)'

#horizontal='temp_mean_annual'
#vertical='rain_mean_annual'

#horizontal='temp_mean_annual'
#vertical='cloudiness'

#horizontal='GDP ($ per capita)'
#vertical='Phones (per 1000)'

div=9

trace0 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'ASIA (EX. NEAR EAST)'],
    y=data_all[vertical][data_all['Region'] == 'ASIA (EX. NEAR EAST)'],
    mode='markers',
    name='ASIA',
    text=data_all['Country'][data_all['Region'] == 'ASIA (EX. NEAR EAST)'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'ASIA (EX. NEAR EAST)'])/div)
    )
)



trace1 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'LATIN AMER. & CARIB'],
    y=data_all[vertical][data_all['Region'] == 'LATIN AMER. & CARIB'],
    mode='markers',
    name='LATIN AMER',
    text=data_all['Country'][data_all['Region'] == 'LATIN AMER. & CARIB'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'LATIN AMER. & CARIB'])/div)
    )
)

trace2 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'SUB-SAHARAN AFRICA'],
    y=data_all[vertical][data_all['Region'] == 'SUB-SAHARAN AFRICA'],
    mode='markers',
    name='AFRICA',
    text=data_all['Country'][data_all['Region'] == 'SUB-SAHARAN AFRICA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'SUB-SAHARAN AFRICA'])/div)
    )
)


trace3 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'C.W. OF IND. STATES'],
    y=data_all[vertical][data_all['Region'] == 'C.W. OF IND. STATES'],
    mode='markers',
    name='Former USSR',
    text=data_all['Country'][data_all['Region'] == 'C.W. OF IND. STATES'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'C.W. OF IND. STATES'])/div)
    )
)


trace4 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'WESTERN EUROPE'],
    y=data_all[vertical][data_all['Region'] == 'WESTERN EUROPE'],
    mode='markers',
    name='WESTERN EUROPE',
    text=data_all['Country'][data_all['Region'] == 'WESTERN EUROPE'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'WESTERN EUROPE'])/div)
    )
)

trace5 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'OCEANIA'],
    y=data_all[vertical][data_all['Region'] == 'OCEANIA'],
    mode='markers',
    name='OCEANIA',
    text=data_all['Country'][data_all['Region'] == 'OCEANIA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'OCEANIA'])/div)
    )
)


trace6 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'NEAR EAST'],
    y=data_all[vertical][data_all['Region'] == 'NEAR EAST'],
    mode='markers',
    name='NEAR EAST',
    text=data_all['Country'][data_all['Region'] == 'NEAR EAST'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'NEAR EAST'])/div)
    )
)


trace7 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'EASTERN EUROPE'],
    y=data_all[vertical][data_all['Region'] == 'EASTERN EUROPE'],
    mode='markers',
    name='EASTERN EUROPE',
    text=data_all['Country'][data_all['Region'] == 'EASTERN EUROPE'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'EASTERN EUROPE'])/div)
    )
)


trace8 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'NORTHERN AFRICA'],
    y=data_all[vertical][data_all['Region'] == 'NORTHERN AFRICA'],
    mode='markers',
    name='NORTHERN AFRICA',
    text=data_all['Country'][data_all['Region'] == 'NORTHERN AFRICA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'NORTHERN AFRICA'])/div)
    )
)


trace9 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'NORTHERN AMERICA'],
    y=data_all[vertical][data_all['Region'] == 'NORTHERN AMERICA'],
    mode='markers',
    name='NORTHERN AMERICA',
    text=data_all['Country'][data_all['Region'] == 'NORTHERN AMERICA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'NORTHERN AMERICA'])/div)
    )
)


trace10 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'BALTICS'],
    y=data_all[vertical][data_all['Region'] == 'BALTICS'],
    mode='markers',
    name='BALTICS',
    text=data_all['Country'][data_all['Region'] == 'BALTICS'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'BALTICS'])/div)
    )
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]

layout = go.Layout(
    title=labels[horizontal]+' vs. '+labels[vertical],
    xaxis=dict(
        title=labels[horizontal],
        titlefont=dict(
            family='Courier New, monospace',
            size=18
          
        )
    ),
    yaxis=dict(
        title=labels[vertical],
        titlefont=dict(
            family='Courier New, monospace',
            size=18
            
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
import numpy as np

init_notebook_mode(connected=True)

#horizontal='Birthrate'
#vertical='Deathrate'

#horizontal='temp_mean_annual'
#vertical='GDP ($ per capita)'

horizontal='temp_mean_annual'
vertical='GDP ($ per capita)'

#horizontal='temp_mean_annual'
#vertical='cloudiness'

#horizontal='GDP ($ per capita)'
#vertical='Phones (per 1000)'

div=8

trace0 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'ASIA (EX. NEAR EAST)'],
    y=data_all[vertical][data_all['Region'] == 'ASIA (EX. NEAR EAST)'],
    mode='markers',
    name='ASIA',
    text=data_all['Country'][data_all['Region'] == 'ASIA (EX. NEAR EAST)'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'ASIA (EX. NEAR EAST)'])/div)
    )
)



trace1 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'LATIN AMER. & CARIB'],
    y=data_all[vertical][data_all['Region'] == 'LATIN AMER. & CARIB'],
    mode='markers',
    name='LATIN AMER',
    text=data_all['Country'][data_all['Region'] == 'LATIN AMER. & CARIB'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'LATIN AMER. & CARIB'])/div)
    )
)

trace2 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'SUB-SAHARAN AFRICA'],
    y=data_all[vertical][data_all['Region'] == 'SUB-SAHARAN AFRICA'],
    mode='markers',
    name='AFRICA',
    text=data_all['Country'][data_all['Region'] == 'SUB-SAHARAN AFRICA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'SUB-SAHARAN AFRICA'])/div)
    )
)


trace3 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'C.W. OF IND. STATES'],
    y=data_all[vertical][data_all['Region'] == 'C.W. OF IND. STATES'],
    mode='markers',
    name='Former USSR',
    text=data_all['Country'][data_all['Region'] == 'C.W. OF IND. STATES'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'C.W. OF IND. STATES'])/div)
    )
)


trace4 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'WESTERN EUROPE'],
    y=data_all[vertical][data_all['Region'] == 'WESTERN EUROPE'],
    mode='markers',
    name='WESTERN EUROPE',
    text=data_all['Country'][data_all['Region'] == 'WESTERN EUROPE'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'WESTERN EUROPE'])/div)
    )
)

trace5 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'OCEANIA'],
    y=data_all[vertical][data_all['Region'] == 'OCEANIA'],
    mode='markers',
    name='OCEANIA',
    text=data_all['Country'][data_all['Region'] == 'OCEANIA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'OCEANIA'])/div)
    )
)


trace6 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'NEAR EAST'],
    y=data_all[vertical][data_all['Region'] == 'NEAR EAST'],
    mode='markers',
    name='NEAR EAST',
    text=data_all['Country'][data_all['Region'] == 'NEAR EAST'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'NEAR EAST'])/div)
    )
)


trace7 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'EASTERN EUROPE'],
    y=data_all[vertical][data_all['Region'] == 'EASTERN EUROPE'],
    mode='markers',
    name='EASTERN EUROPE',
    text=data_all['Country'][data_all['Region'] == 'EASTERN EUROPE'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'EASTERN EUROPE'])/div)
    )
)


trace8 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'NORTHERN AFRICA'],
    y=data_all[vertical][data_all['Region'] == 'NORTHERN AFRICA'],
    mode='markers',
    name='NORTHERN AFRICA',
    text=data_all['Country'][data_all['Region'] == 'NORTHERN AFRICA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'NORTHERN AFRICA'])/div)
    )
)


trace9 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'NORTHERN AMERICA'],
    y=data_all[vertical][data_all['Region'] == 'NORTHERN AMERICA'],
    mode='markers',
    name='NORTHERN AMERICA',
    text=data_all['Country'][data_all['Region'] == 'NORTHERN AMERICA'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'NORTHERN AMERICA'])/div)
    )
)


trace10 = go.Scatter(
    x=data_all[horizontal][data_all['Region'] == 'BALTICS'],
    y=data_all[vertical][data_all['Region'] == 'BALTICS'],
    mode='markers',
    name='BALTICS',
    text=data_all['Country'][data_all['Region'] == 'BALTICS'],
    marker=dict(
    size=list(np.cbrt(data_all['Population'][data_all['Region'] == 'BALTICS'])/div)
    )
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]

layout = go.Layout(
    title=labels[horizontal]+' vs. '+labels[vertical],
    xaxis=dict(
        title=labels[horizontal],
        titlefont=dict(
            family='Courier New, monospace',
            size=18
          
        )
    ),
    yaxis=dict(
        title=labels[vertical],
        titlefont=dict(
            family='Courier New, monospace',
            size=18
            
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)