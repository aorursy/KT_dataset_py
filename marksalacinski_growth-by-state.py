import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import HTML, Image



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px



# Confirm input files have not changed in this dataset

for dirname, _, filenames in os.walk('/kaggle/input'):

    assert len(filenames) in [0, 3]



# Read data

n_deaths = pd.read_csv('/kaggle/input/coronavirus-4th-mar-2020-johns-hopkins-university/time_series_2019-ncov-Deaths.csv')

n_recov = pd.read_csv('/kaggle/input/coronavirus-4th-mar-2020-johns-hopkins-university/time_series_2019-ncov-Recovered.csv')

n_conf = pd.read_csv('/kaggle/input/coronavirus-4th-mar-2020-johns-hopkins-university/time_series_2019-ncov-Confirmed.csv')



# Melt dates from columns into rows

id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long']

n_deaths = n_deaths.melt(id_vars=id_vars, value_vars=None, var_name='Date', value_name='Deaths')

n_recov = n_recov.melt(id_vars=id_vars, value_vars=None, var_name='Date', value_name='Recovered')

n_conf = n_conf.melt(id_vars=id_vars, value_vars=None, var_name='Date', value_name='Confirmed')

print(f"DataFrame shapes -- Deaths: {n_deaths.shape}    Recovered: {n_recov.shape}    Confirmed: {n_conf.shape}")



# Join together (and check for unexpected NAs and misaligned dates)

assert n_deaths.Date.unique().tolist() == n_recov.Date.unique().tolist() == n_conf.Date.unique().tolist() 

join_cols = id_vars + ['Date']

df = pd.merge(n_conf, n_deaths, how='outer', on=join_cols)

df = pd.merge(df, n_recov, how='outer', on=join_cols)

df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

assert df.shape[0] == n_deaths.shape[0] == n_recov.shape[0] == n_conf.shape[0]

assert 0 == df.drop(columns=['Province/State']).isna().sum().sum()

print(f"Combined 'df' shape: {df.shape}")

df.head()
# This dataset contains city-level data in the United States up until March 10th, after which 

# all data gets aggregated to the state-level and the city-level figures reset to zero.

# We'll want to extract the State from each city-level figure, conform to US State abbr. to name, and group-sum by date+state



# First, filter on United States only and remove cruise ships

us = df[df['Country/Region']=='US']

us = us[~us['Province/State'].isin(["Diamond Princess", "Grand Princess"])]



# Find cities by the presence of a comma "," and normalize

STATE_TO_STATE_ABBR = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC', 'Marshall Islands': 'MH', 'Armed Forces Africa': 'AE', 'Armed Forces Americas': 'AA', 'Armed Forces Canada': 'AE', 'Armed Forces Europe': 'AE', 'Armed Forces Middle East': 'AE', 'Armed Forces Pacific': 'AP', }

STATE_ABBR_TO_STATE = {v:k for k,v in STATE_TO_STATE_ABBR.items()}

city_to_state_remapper = {

    city: STATE_ABBR_TO_STATE.get(city.split(', ')[1].replace('.','').strip(), None)

    for city in us.loc[us['Province/State'].str.contains(','), 'Province/State'].unique().tolist()

}

us['State'] = us['Province/State'].map(city_to_state_remapper).fillna(us['Province/State'])



# Group and sum to aggregate back to one row per region+date

us = (us

      .drop(columns=['Country/Region', 'Province/State', 'Lat', 'Long'])

      .groupby(['State', 'Date'])

      .sum()

      .reset_index()

     )





# Produce a set of delta values

# NOTE: These are not guaranteed to be day-to-day, sometimes there are date gaps

us['Delta Confirmed'] = us['Confirmed'].copy()

us['Delta Deaths'] = us['Deaths'].copy()

us['Delta Recovered'] = us['Recovered'].copy()

cols_delta = ['Delta Confirmed', 'Delta Deaths', 'Delta Recovered']

us[cols_delta] = us[cols_delta] - us.groupby('State')[cols_delta].shift(1).fillna(0)

us[cols_delta] = us[cols_delta].astype(int)



us.tail(5)
px.line(us[us.Date > '2020-03-01'], 

        x="Date", 

        y="Confirmed",

        log_y=True,

        color="State",

#         line_group="State", 

#         hover_name="State"

        title="Logarithmic Growth of Confirmed Cases, by State",

       )
us['StateAbbr'] = us['State'].map(STATE_TO_STATE_ABBR)

us['DateStr'] = us['Date'].dt.strftime('%Y-%m-%d')

fig = px.choropleth(us, 

                    scope="usa",

                    locationmode='USA-states',

                    locations="StateAbbr", 

                    color="Confirmed", 

                    hover_name="State",

                    hover_data=["Confirmed", "Deaths", "Recovered"] + cols_delta,

                    animation_frame="DateStr", 

                    title="United States Confirmed Cases<BR>*Use slider at bottom to cycle dates",

                    color_continuous_scale=px.colors.sequential.Oranges,

#                     range_color=[0,us['Confirmed'].max()]

                   )

fig.show()
# Calculate week-over-week growth and plot that

# TODO