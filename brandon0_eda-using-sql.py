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
from sqlalchemy import create_engine



import plotly.express as px

import plotly.io as pio





pio.templates.default = "plotly_dark"





import json
filename_train = '/kaggle/input/covid19-global-forecasting-week-4/train.csv'





train_df = pd.read_csv(filename_train)



engine = create_engine('sqlite://', echo=False)

train_df.to_sql('train', con=engine)
table = 'train'



query = """

    SELECT Country_Region AS country,

            Date,

            SUM(ConfirmedCases) AS confirmed,

            SUM(Fatalities) AS death

    FROM {0}

    GROUP BY Date, Country_Region

    ORDER BY Country_Region, Date

""".format(table)



print(query)



sum_world = pd.read_sql(query, engine)



sum_world['size_confirmed'] = sum_world['confirmed'].pow(0.3)

sum_world['size_death'] = sum_world['death'].pow(0.3)



sum_world.head()
fig = px.scatter_geo(sum_world, locations='country',

                    locationmode='country names', color='confirmed',

                    size='size_confirmed', hover_name='country', range_color=[1, 1500],

                    projection='natural earth', animation_frame='Date',

                    title='Accumulated Confirmed Cases Over Time', color_continuous_scale="portland")



fig.show()


fig = px.line(sum_world,

               x='Date', y='confirmed',

               color='country',

               title='Accumulated Confirmed Cases Over The World')



fig.show()
fig = px.scatter_geo(sum_world, locations='country',

                    locationmode='country names', color='death',

                    size='size_confirmed', hover_name='country', range_color=[1, 1500],

                    projection='natural earth', animation_frame='Date',

                    title='COVID-19: Death Over Time', color_continuous_scale="portland")



fig.show()


fig = px.line(sum_world,

               x='Date', y='death',

               color='country',

               title='Accumulated Death Toll Over The World')



fig.show()
table = 'train'

country = 'US'



query = """

    SELECT t.Province_State AS state,

            t.Date,

            SUM(ConfirmedCases) AS confirmed,

            SUM(Fatalities) AS death

    FROM {0} AS t

    GROUP BY t.Date, t.Province_State HAVING t.Country_Region = \'{1}\'

    ORDER BY t.Province_State, t.Date

""".format(table, country)



print(query)



sum_us = pd.read_sql(query, engine)



sum_us.head()


fig = px.line(sum_us,

               x='Date', y='confirmed',

               color='state',

               title='Accumulated Confirmed Cases in the US')



fig.show()


fig = px.line(sum_us,

               x='Date', y='death',

               color='state',

               title='Accumulated Confirmed Cases in the US')



fig.show()
yesterday = sum_us['Date'].max()



sum_yesterday_us = sum_us[sum_us['Date'] == yesterday]



sum_yesterday_us.head()
us_states_json = json.loads("""

{

    "AL": "Alabama",

    "AK": "Alaska",

    "AS": "American Samoa",

    "AZ": "Arizona",

    "AR": "Arkansas",

    "CA": "California",

    "CO": "Colorado",

    "CT": "Connecticut",

    "DE": "Delaware",

    "DC": "District Of Columbia",

    "FM": "Federated States Of Micronesia",

    "FL": "Florida",

    "GA": "Georgia",

    "GU": "Guam",

    "HI": "Hawaii",

    "ID": "Idaho",

    "IL": "Illinois",

    "IN": "Indiana",

    "IA": "Iowa",

    "KS": "Kansas",

    "KY": "Kentucky",

    "LA": "Louisiana",

    "ME": "Maine",

    "MH": "Marshall Islands",

    "MD": "Maryland",

    "MA": "Massachusetts",

    "MI": "Michigan",

    "MN": "Minnesota",

    "MS": "Mississippi",

    "MO": "Missouri",

    "MT": "Montana",

    "NE": "Nebraska",

    "NV": "Nevada",

    "NH": "New Hampshire",

    "NJ": "New Jersey",

    "NM": "New Mexico",

    "NY": "New York",

    "NC": "North Carolina",

    "ND": "North Dakota",

    "MP": "Northern Mariana Islands",

    "OH": "Ohio",

    "OK": "Oklahoma",

    "OR": "Oregon",

    "PW": "Palau",

    "PA": "Pennsylvania",

    "PR": "Puerto Rico",

    "RI": "Rhode Island",

    "SC": "South Carolina",

    "SD": "South Dakota",

    "TN": "Tennessee",

    "TX": "Texas",

    "UT": "Utah",

    "VT": "Vermont",

    "VI": "Virgin Islands",

    "VA": "Virginia",

    "WA": "Washington",

    "WV": "West Virginia",

    "WI": "Wisconsin",

    "WY": "Wyoming"

} 

""")

    

# switch key/value from code/state to state/code.

us_states = {state: abbrev for abbrev, state in us_states_json.items()}

    

    

# add state code column

sum_yesterday_us['code'] = sum_yesterday_us['state'].map(us_states)
fig = px.choropleth(sum_yesterday_us, locations='code',

                   locationmode='USA-states', color='confirmed',

                   hover_name='state', range_color=[1, 10000],

                   scope='usa',

                    color_continuous_scale='portland',

                   title='Confirmed Cases on {}'.format(yesterday))



fig.show()
fig = px.choropleth(sum_yesterday_us, locations='code',

                   locationmode='USA-states', color='death',

                   hover_name='state', range_color=[1, 1000],

                   scope='usa',

                    color_continuous_scale='portland',

                   title='Death Toll on {}'.format(yesterday))



fig.show()
table = 'train'



query = """

    WITH summary AS (

        SELECT t.*,

                SUM(t.ConfirmedCases) AS confirmed,

                SUM(t.Fatalities) AS death

        FROM {0} AS t

        GROUP BY t.Date, t.Country_Region

    )

    

    SELECT s.Country_Region AS country,

            s.Date,

            s.confirmed,

            s.death,

            s.confirmed - LAG(s.confirmed) OVER (PARTITION BY s.Country_Region ORDER BY s.Date) AS daily_confirmed,

            s.death - LAG(s.death) OVER (PARTITION BY s.Country_Region ORDER BY s.Date) AS daily_death

    FROM summary AS s

    ORDER BY s.Country_Region

""".format(table)



print(query)



daily_world = pd.read_sql(query, engine)



daily_world['size_daily_confirmed'] = daily_world['daily_confirmed'].pow(0.3)

daily_world['size_daily_death'] = daily_world['daily_death'].pow(0.3)



daily_world = daily_world.fillna(0)





daily_world.head()
fig = px.scatter_geo(daily_world, locations='country',

                    locationmode='country names', color='daily_confirmed',

                    size='size_daily_confirmed', hover_name='country', range_color=[0, 1500],

                    projection='natural earth', animation_frame='Date',

                    title='Daily New Confirmed Cases', color_continuous_scale="portland")



fig.show()


fig = px.line(daily_world, 

              x='Date', y='daily_confirmed',

             color='country',

             title='Daily Confirmed Cases Over Time')



fig.show()


fig = px.line(daily_world, 

              x='Date', y='daily_death',

             color='country',

             title='Daily Death Toll Over Time')



fig.show()
table = 'train'

country = 'US'



query = """

    WITH us AS (

        SELECT t.*

        FROM {0} AS t

        WHERE t.Country_Region = \'{1}\'

    )

    

    SELECT u.Country_Region AS country,

            u.Province_State AS state,

            u.Date,

            u.ConfirmedCases AS confirmed,

            u.Fatalities AS death,

            u.ConfirmedCases - LAG(u.ConfirmedCases) OVER (PARTITION BY u.Province_State ORDER BY u.Date) AS daily_confirmed,

            u.Fatalities - LAG(u.Fatalities) OVER (PARTITION BY u.Province_State ORDER BY u.Date) AS daily_death

    FROM us AS u

""".format(table, country)



print(query)



daily_us = pd.read_sql(query, engine)



daily_us.head()


fig = px.line(daily_us, 

              x='Date', y='daily_confirmed',

             color='state',

             title='Daily New Confirmed Cases Over Time in US')



fig.show()
# add state code column

daily_us['code'] = daily_us['state'].map(us_states)
yesterday = daily_us['Date'].max()

daily_yesterday_us = daily_us[daily_us['Date'] == yesterday]



fig = px.choropleth(daily_yesterday_us, locations='code',

                   locationmode='USA-states', color='daily_confirmed',

                   hover_name='state', range_color=[1, 1000],

                   scope='usa',

                    color_continuous_scale='portland',

                   title='Daily Confrimed Case on {}'.format(yesterday))



fig.show()


fig = px.line(daily_us, 

              x='Date', y='daily_death',

             color='state',

             title='Daily Death Toll Over Time in US')



fig.show()
yesterday = daily_us['Date'].max()

daily_yesterday_us = daily_us[daily_us['Date'] == yesterday]



fig = px.choropleth(daily_yesterday_us, locations='code',

                   locationmode='USA-states', color='daily_death',

                   hover_name='state', range_color=[1, 100],

                   scope='usa',

                    color_continuous_scale='portland',

                   title='Daily Death Toll on {}'.format(yesterday))



fig.show()