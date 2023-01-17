import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
# Load csv into dataframe

df = pd.read_csv('../input/pollution_us_2000_2016.csv')
# Looking the data

df.head()
# Checking types of each column

df.dtypes
# Checking the amount of null in every column

df.isnull().sum()
# Assigning abbreviatures

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



df['State Abbrev'] = df['State'].map(us_state_abbrev)
# Grouping data by state and time

df['Date Local'] = pd.to_datetime(df['Date Local'])

df_groupby_date = df.groupby(['State', 'State Abbrev', df['Date Local'].dt.year])['NO2 AQI'].mean()
# Converting serie to dataframe

df_groupped = df_groupby_date.to_frame()

df_groupped = df_groupped.reset_index()
# Selecting data

df_groupped['Date Local'] = df_groupped['Date Local'].astype(str)

df_groupped_2016 = df_groupped[(df_groupped['Date Local'] == '2016')]
from plotly.offline import init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\

            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



data = [dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = df_groupped_2016['State Abbrev'],

        z = df_groupped_2016['NO2 AQI'],

        locationmode = 'USA-states',

        text = df_groupped_2016['State'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "NO2 AQI")

        )]



layout = dict(

        title = 'NO2 in US 2016',

        geo = dict(

            scope='usa',

            projection= dict(type='albers usa'),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

        )

    

fig = dict(data=data, layout=layout)

iplot(fig, filename='pollution')