# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import altair as alt
df = pd.read_csv('/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv',parse_dates=['Year'])

df['Annual_Emissions'] = df['Annual CO₂ emissions (tonnes )']/1000000000

df.drop(['Annual CO₂ emissions (tonnes )', 'Code'],axis=1, inplace=True)

df['DT_Year']=df.Year.dt.year

df.head()
# Create separate dataframes for each continent

mideast = df[df['Entity'] == 'Middle East']

africa = df[df['Entity'] == 'Africa']

americas = df[df['Entity'] == 'Americas (other)']

apac = df[df['Entity'] == 'Asia and Pacific (other)']

euro = df[df['Entity'] == 'EU-28']

europe = df[df['Entity'] == 'Europe (other)']

usa = df[df['Entity'] == 'United States']

china = df[df['Entity'] == 'China']

transport = df[df['Entity'] == 'International transport']

world = df[df['Entity'] == 'World']



conts = [africa, americas, apac, euro, europe, mideast, usa, china, transport, world]

continents = pd.concat(conts)

print(continents['Entity'].unique())
g20 = df.query("Entity == ['Argentina', 'Australia', 'Brazil', 'Canada', 'Saudi Arabia','China', 'France', 'Germany', 'India', 'United States','Indonesia', 'Italy', 'Japan', 'Mexico', 'Russia','South Africa', 'South Korea', 'Turkey', 'United Kingdom', 'Spain']")

print(g20['Entity'].unique())
multi = alt.selection_multi(on='mouseover')



alt.Chart(continents).mark_area(opacity=0.5, line=True).encode(

        x = alt.X('year(Year):T'),

        y = alt.Y('Annual_Emissions', title='Annual Emissions (in billion tonnes)'),

        color = alt.condition(multi,'Entity', alt.value('lightgray'), title='Region'),

        tooltip = [alt.Tooltip('Year:T', format='%Y'),

                   'Entity', 'Annual_Emissions']

).properties(

        width = 600,

        height = 400,

        title='Total Annual CO2 emissions, by region',

        selection = multi

)
# Year slider

year_slider = alt.binding_range(min=1751, max=2017, step=1)

slider = alt.selection_single(bind=year_slider, fields=['DT_Year'], name='Select', init={'DT_Year':2017})



# Single selection

click = alt.selection_single()



# Main Chart



alt.Chart(continents).mark_bar().encode(

        x = alt.X('Entity', title='Country/Region'),

        y = alt.Y('Annual_Emissions', title='Annual Emissions (in billion tonnes)'),

        color = alt.condition(click, 'Entity', alt.value('lightgray'),title='Country/Region'),

        tooltip= ['Annual_Emissions']

).properties(

    width=600,

    height=400,

    title='Total Annual CO2 emissions, by region',

    selection=slider

).transform_filter(

    slider

).add_selection(

    click

)
# G20 plot



# Year slider

year_slider = alt.binding_range(min=1751, max=2017, step=1)

slider = alt.selection_single(bind=year_slider, fields=['DT_Year'], name='Select', init={'DT_Year':2017})



# Single selection

click = alt.selection_single()



# Main Chart



alt.Chart(g20).mark_bar().encode(

        x = alt.X('Entity', title='Countries'),

        y = alt.Y('Annual_Emissions', title='Annual Emissions (in billion tonnes)'),

        color = alt.condition(click, 'Entity', alt.value('lightgray'), title='Countries'),

        tooltip= ['Annual_Emissions']

).properties(

    width=600,

    height=400,

    title='Total Annual CO2 emissions, by G20 countries',

    selection=slider

).transform_filter(

    slider

).add_selection(

    click

)
country_dr = alt.binding_select(options=[None,'Argentina', 'Australia', 'Brazil', 'Canada', 'Saudi Arabia','China', 

                                         'France', 'Germany', 'India', 'United States','Indonesia', 'Italy', 

                                         'Japan', 'Mexico', 'Russia','South Africa', 'South Korea', 'Turkey', 

                                         'United Kingdom', 'Spain'])

country_sl = alt.selection_single(fields=['Entity'], bind=country_dr, name="Country", clear='click')





alt.Chart(g20).mark_area(opacity=0.5, line=True).encode(

        x = alt.X('year(Year):T', title='Year'),

        y = alt.Y('Annual_Emissions:Q', title='Annual Emissions (in billion tonnes)'),

        color = ('Entity'),

        tooltip = [alt.Tooltip('Year:T', format='%Y'),

                   'Entity', 'Annual_Emissions']

).properties(

        width = 600,

        height = 400,

        title = 'Annual CO2 Emissions, by G20 countries'

).add_selection(

    country_sl

).transform_filter(

    country_sl

)