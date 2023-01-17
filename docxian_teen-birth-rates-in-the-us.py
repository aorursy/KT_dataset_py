import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load csv file

df = pd.read_csv('../input/us-teen-birth-rates-ages-1519-20032018/NCHS_-_Teen_Birth_Rates_for_Age_Group_15-19_in_the_United_States_by_County.csv')
# first glance

df.head(25)
# we have years from 2003..2018 available

df.Year.describe()
# filter year 2018

df_2018 = df[df.Year==2018]
# aggregation by state

df_2018_state = df_2018.groupby('State', as_index=False).agg(

    min_birth_rate = pd.NamedAgg(column='Birth Rate', aggfunc=min),

    max_birth_rate = pd.NamedAgg(column='Birth Rate', aggfunc=max),

    mean_birth_rate = pd.NamedAgg(column='Birth Rate', aggfunc=np.mean))



df_2018_state
plt.figure(figsize=(14,8))

sns.barplot(x=df_2018_state.State, y=df_2018_state.mean_birth_rate)

plt.xticks(rotation=90)

plt.grid()

plt.title('Mean Teen Birth Rate 2018')

plt.show()
# first we have to translate the state names to abbreviations

# we use a dictionary for this (many thanks to https://gist.github.com/rogerallen/1583593 for providing this already!)

state_abbreviations = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'American Samoa': 'AS',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Guam': 'GU',

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

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY'

}
df_2018_state['StateCode'] = df_2018_state['State'].map(state_abbreviations)

df_2018_state
# now we can plot a map

fig = px.choropleth(df_2018_state,

                    locationmode='USA-states',

                    scope='usa',

                    locations='StateCode',

                    color='mean_birth_rate',

                    hover_name='State',

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
df_by_state = df.groupby(['Year','State'], as_index=False).agg(

    min_birth_rate = pd.NamedAgg(column='Birth Rate', aggfunc=min),

    max_birth_rate = pd.NamedAgg(column='Birth Rate', aggfunc=max),

    mean_birth_rate = pd.NamedAgg(column='Birth Rate', aggfunc=np.mean))



df_by_state
# select a specific state

select_state='Alabama'

df_by_state_sel = df_by_state[df_by_state.State==select_state]

df_by_state_sel
plt.figure(figsize=(7,4))

plt.plot(df_by_state_sel.Year, df_by_state_sel.mean_birth_rate)

plt.grid()

plt.title('Mean Birth Rate - Alabama')

plt.show()
# select another state

select_state='New York'

df_by_state_sel = df_by_state[df_by_state.State==select_state]

df_by_state_sel
plt.figure(figsize=(7,4))

plt.plot(df_by_state_sel.Year, df_by_state_sel.mean_birth_rate)

plt.grid()

plt.title('Mean Birth Rate - New York')

plt.show()
plt.figure(figsize=(12,8))

sns.lineplot('Year','mean_birth_rate', hue='State', data=df_by_state)

plt.grid()

plt.title('Mean Birth Rate')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()