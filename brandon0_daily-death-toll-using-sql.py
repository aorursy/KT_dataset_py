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
import plotly.express as px

import plotly.io as pio



from collections import defaultdict



import json





pio.templates.default = "plotly_dark"
filename_train = '/kaggle/input/covid19-global-forecasting-week-3/train.csv'

filename_test = '/kaggle/input/covid19-global-forecasting-week-3/test.csv'
train = pd.read_csv(filename_train)

test = pd.read_csv(filename_test)
display(train.head())

display(test.head())
from sqlalchemy import create_engine

engine = create_engine('sqlite://', echo=False)



train.to_sql('train', con=engine)

test.to_sql('test', con=engine)



print(engine.table_names())
table = "train"

query = "SELECT Date, SUM(ConfirmedCases) as confirmed FROM {} GROUP BY Date".format(table)

df = pd.read_sql(query, engine)



df.head()
fig = px.line(df, x='Date', y='confirmed',

              title="Worldwide Confirmed Cases Over Time")

fig.show()



fig = px.line(df, x='Date', y='confirmed',

              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time",

              log_y=True)



fig.show()
def query_country (country, table):

    query = """

        WITH tmp AS

        (

            SELECT * FROM {} WHERE Country_Region=\'{}\'

        )

        SELECT Date, SUM(ConfirmedCases) AS confirmed

        FROM tmp

        GROUP BY Date

            """.format(table, country)



    print(query)

    df = pd.read_sql(query, engine)

    # df.head()

    return df







results = defaultdict()



table = 'train'

country = 'China'

results[country] = query_country(country, table)

######

table = 'train'

country = 'Italy'

results[country] = query_country(country, table)

######

table = 'train'

country = 'US'

results[country] = query_country(country, table)

######

table = 'train'

country = 'Korea, South'

results[country] = query_country(country, table)
table = 'train'

excluded_countries = ('China', 'Italy', 'US', 'Korea, South')



query = """

    WITH tmp AS

    (

        SELECT * FROM {} WHERE Country_Region NOT IN {}

    )

    SELECT Date, SUM(ConfirmedCases) AS confirmed

    FROM tmp

    GROUP BY Date

""".format(table, excluded_countries)



print(query)



df = pd.read_sql(query, engine)



results['Rest of the World'] = df
colors = ('#F61067', '#91C4F2', '#6F2DBD', '#00FF00', '#FFDF64')



for c, (country, df) in zip(colors, results.items()):

    fig = px.line(results[country], x='Date', y='confirmed',

                 title="Confirmed Cases in {} Over Time".format(country),

                 color_discrete_sequence=[c],

                 height=500)

    fig.show()
table = "train"



query = """

    SELECT t.Date, t.Country_Region AS country, SUM(t.ConfirmedCases) as confirmed

    FROM {0} AS t

    INNER JOIN (

        SELECT max(Date) AS MaxDate, Country_Region FROM {0} GROUP BY Country_Region

    ) tmp

    ON tmp.MaxDate=t.Date and tmp.Country_Region = t.Country_Region

    GROUP BY t.Country_Region

    """.format(table)



print(query)



latest_grouped = pd.read_sql(query, engine)

latest_grouped.head()
fig = px.choropleth(latest_grouped, locations='country',

                   locationmode='country names', color='confirmed',

                   hover_name='country', range_color=[1, 5000],

                   color_continuous_scale='peach',

                   title='Countries with Confirmed Cases')



fig.show()
table = "train"



europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

               'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

               'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',

               'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])





europe_grouped_latest  = latest_grouped[latest_grouped['country'].isin(europe)]

europe_grouped_latest.head()
fig = px.choropleth(europe_grouped_latest, locations='country',

                   locationmode='country names', color='confirmed',

                   hover_name='country', range_color=[1, 2000],

                   color_continuous_scale='portland', 

                    title='European Countries with Confirmed Cases', scope='europe', height=800)



fig.show()
fig = px.bar(latest_grouped.sort_values('confirmed', ascending=False)[:20][::-1], 

             x='confirmed', y='country',

             title='Confirmed Cases Worldwide', text='confirmed', height=1000, orientation='h')

fig.show()
fig = px.bar(europe_grouped_latest.sort_values('confirmed', ascending=False)[:10][::-1], 

             x='confirmed', y='country', color_discrete_sequence=['#84DCC6'],

             title='Confirmed Cases in Europe', text='confirmed', orientation='h')

fig.show()
table = "train"

country = 'US'



query = """

    SELECT t.Date, t.Province_State AS state, t.ConfirmedCases AS confirmed

    FROM {0} AS t

    INNER JOIN

    (

        SELECT MAX(Date) as MaxDate, Province_State, Country_Region 

        FROM {0}

        GROUP BY Province_State, Country_Region

    ) AS tmp

    ON tmp.MaxDate = t.Date AND 

         tmp.Province_State = t.Province_State AND

         tmp.Country_Region = t.Country_Region

    WHERE t.Country_Region = \'{1}\'

    ORDER BY t.ConfirmedCases DESC

""".format(table, country)



print(query)



usa_latest = pd.read_sql(query, engine)

usa_latest.head()
fig = px.bar(usa_latest[:10][::-1], 

            x = 'confirmed', y='state', color_discrete_sequence=['#D63230'],

            title='Confirmed Cases in USA', text='confirmed', orientation='h')



fig.show()
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

usa_latest['code'] = usa_latest['state'].map(us_states)
fig = px.choropleth(usa_latest, locations='code',

                   locationmode='USA-states', color='confirmed',

                    hover_name='state', range_color=[1, 10000],

                   scope='usa')



fig.show()
table = 'train'



query = """

    SELECT Date, Country_Region AS country, SUM(ConfirmedCases) AS confirmed, SUM(Fatalities) AS deaths

    FROM {}

    GROUP BY Date, Country_Region

""".format(table)



print(query)



formated_gdf = pd.read_sql(query, engine)





formated_gdf['size'] = formated_gdf['confirmed'].pow(0.3)



display(formated_gdf)

fig = px.scatter_geo(formated_gdf, locations='country',

                    locationmode='country names', color='confirmed',

                    size='size', hover_name='country', range_color=[0, 1500],

                    projection='natural earth', animation_frame='Date',

                    title='COVID-19: Spread Over Time', color_continuous_scale="portland")



fig.show()
fig = px.scatter_geo(formated_gdf, locations='country',

                    locationmode='country names', color='deaths',

                    size='size', hover_name='country', range_color=[0, 1500],

                    projection='natural earth', animation_frame='Date',

                    title='COVID-19: Death Over Time', color_continuous_scale="portland")



fig.show()
table = 'train'



query = """

    WITH country_table AS (

        SELECT Date, 

                Country_Region As country,

                SUM(ConfirmedCases) AS confirmed, 

                SUM(Fatalities) AS deaths

        FROM {0}

        GROUP BY Date, Country_Region

    )

    

    

    SELECT *,

            confirmed - LAG(confirmed) OVER country_window AS new_confirmed_cases,

            deaths - LAG(deaths) OVER country_window AS new_death_cases

    FROM country_table

    

    

    WINDOW country_window AS (

        PARTITION BY country ORDER BY Date

    )

""".format(table)



print(query)



new_cases_world = pd.read_sql(query, engine)



new_cases_world['new_confirmed_cases'].fillna(0, inplace=True)

new_cases_world['new_death_cases'].fillna(0, inplace=True)
countries = ('US', 'Italy', 'Spain', 'Korea, South', 'United Kingdom')



fig = px.line(new_cases_world[new_cases_world['country'].isin(countries)], 

              x='Date', y='new_confirmed_cases',

             color='country',

             title='Daily Confirmed Cases in {} Over Time'.format(countries))



fig.show()
countries = ('US', 'Italy', 'Spain', 'Korea, South', 'United Kingdom')



fig = px.line(new_cases_world[new_cases_world['country'].isin(countries)], 

              x='Date', y='new_death_cases',

             color='country',

             title='Daily Death Cases in {} Over Time'.format(countries))



fig.show()
new_cases_world['size_new_confirmed_cases'] = new_cases_world['new_confirmed_cases'].pow(0.3) 



display(new_cases_world[new_cases_world['size_new_confirmed_cases'].isnull()])



new_cases_world['size_new_confirmed_cases'].fillna(0, inplace=True)







new_cases_world['size_new_death_cases'] = new_cases_world['new_death_cases'].pow(0.3) 



display(new_cases_world[new_cases_world['size_new_death_cases'].isnull()])



new_cases_world['size_new_death_cases'].fillna(0, inplace=True)



fig = px.scatter_geo(new_cases_world, locations='country',

                   locationmode='country names', color='new_confirmed_cases',

                    size='size_new_confirmed_cases',

                   hover_name='new_confirmed_cases', range_color= [0, 1500],

                     projection='natural earth', animation_frame='Date',

                    title='Countries with Daily Confirmed Cases', color_continuous_scale="portland")



fig.show()


fig = px.scatter_geo(new_cases_world, locations='country',

                   locationmode='country names', color='new_death_cases',

                    size='size_new_death_cases',

                   hover_name='new_death_cases', range_color= [0, 500],

                     projection='natural earth', animation_frame='Date',

                    title='Countries with Daily Confirmed Cases', color_continuous_scale="portland")



fig.show()
table = 'train'

country = 'US'



query = """

    SELECT Date, Province_State AS state, Country_Region AS country, ConfirmedCases AS confirmed, ConfirmedCases AS death

    FROM {0}

    WHERE Country_Region = \'{1}\'

""".format(table, country)



print(query)



us_df = pd.read_sql(query, engine)


fig = px.line(us_df,

             x='Date', y='confirmed',

             color='state')



fig.show()


fig = px.line(us_df,

             x='Date', y='death',

             color='state')



fig.show()
table = 'train'

country = 'US'



query = """

    WITH us_table AS (

        SELECT t.*

        FROM {0} AS t

        WHERE t.Country_Region = \'{1}\'

    )

    

    SELECT u.*,

            u.ConfirmedCases - LAG(u.ConfirmedCases) OVER (PARTITION BY u.Province_State ORDER BY u.Date) AS new_confirmed,

            u.Fatalities - LAG(u.Fatalities) OVER (PARTITION BY u.Province_State ORDER BY u.Date) AS new_death

    FROM us_table AS u

""".format(table, country)



print(query)



daily_us = pd.read_sql(query, engine)



daily_us.head()
fig = px.line(daily_us,

             x='Date', y='new_confirmed',

             color='Province_State',

             title='Daily New Confirmed Cases in US')



fig.show()
fig = px.line(daily_us,

             x='Date', y='new_death',

             color='Province_State',

             title='Daily New Confirmed Cases in US')



fig.show()