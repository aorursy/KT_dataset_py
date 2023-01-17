import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
ci = pd.read_csv('../input/farmers-markets-in-the-united-states/wiki_county_info.csv')

fm = pd.read_csv('../input/farmers-markets-in-the-united-states/farmers_markets_from_usda.csv')
ci.head(2)
ci.info()
ci.drop(['number','county'],axis=1,inplace=True)
print(f'number of rows in ci: {len(ci)}\n')

print(f'number of null values by column in ci: \n\n{ci.isnull().sum()}')
rows_to_drop = [i for i in range(len(ci)) if ci.iloc[i].isnull().sum()>0]

for i in rows_to_drop:

    ci.drop(i,axis=0,inplace=True)

ci.isnull().sum()
ci.info()
ci.head(5)
def get_value(x):

    return int(''.join(x.split('$')[1].split(',')))

def get_number(x):

    return int(''.join(x.split(',')))
for i in ['per capita income','median household income','median family income']:

    ci[i] = ci[i].apply(lambda x:get_value(x))



for i in ['population','number of households']:

    ci[i] = ci[i].apply(lambda x:get_number(x))
ci.info()
ci.head()
fm.head(3)
fm.info()
fm = fm[['MarketName','State','x','y','updateTime']]
fm.head(5)
print(f'number of rows in ci: {len(fm)}\n')

print(f'number of null values by column in ci: \n\n{fm.isnull().sum()}')
sns.heatmap(fm.isnull())
fm.dropna(subset=['x','y'],axis=0,inplace=True)

fm.isnull().sum()
fm.head()
fm.info()
def get_year(x):

    return pd.to_datetime(x).year



fm['year updated'] = fm['updateTime'].apply(lambda x:get_year(x))

fm['details'] = 'State: '+fm['State']+' --- '+'Name: '+fm['MarketName']

fm.drop('updateTime',axis=1,inplace=True)
fm.head()
ci.head()
no_markets_per_state = pd.DataFrame(fm['State'].value_counts())

no_markets_per_state.columns = ['no. farmers markets']

no_markets_per_state.sort_values('no. farmers markets').tail()
state_level = ci[['State','per capita income','median household income','median family income','population','number of households']].groupby('State').mean()
markets= pd.concat([state_level,no_markets_per_state], axis=1)



markets['state'] = markets.index



markets.drop('median family income',axis = 1, inplace = True) # Remove median family income as we will use per capita income

markets.drop(index = 'Virgin Islands', inplace = True) # remove the duplicate row for Virgin Islands
for i in markets.drop('state',axis=1).columns:

    markets[i].fillna(markets[i].mean(),inplace=True)
markets.isnull().sum()
markets.head()
state_income = ci[['State','per capita income']] 

av_state_per_capita_income = state_income.groupby('State').mean()

statenames = list(av_state_per_capita_income.index)

statenames[48] = 'Virgin Islands'



states = {'Alaska': 'AK',

 'Alabama': 'AL',

 'Arkansas': 'AR',

 'American Samoa': 'AS',

 'Arizona': 'AZ',

 'California': 'CA',

 'Colorado': 'CO',

 'Connecticut': 'CT',

 'District of Columbia': 'DC',

 'Delaware': 'DE',

 'Florida': 'FL',

 'Georgia': 'GA',

 'Guam': 'GU',

 'Hawaii': 'HI',

 'Iowa': 'IA',

 'Idaho': 'ID',

 'Illinois': 'IL',

 'Indiana': 'IN',

 'Kansas': 'KS',

 'Kentucky': 'KY',

 'Louisiana': 'LA',

 'Massachusetts': 'MA',

 'Maryland': 'MD',

 'Maine': 'ME',

 'Michigan': 'MI',

 'Minnesota': 'MN',

 'Missouri': 'MO',

 'Northern Mariana Islands': 'MP',

 'Mississippi': 'MS',

 'Montana': 'MT',

 'North Carolina': 'NC',

 'North Dakota': 'ND',

 'Nebraska': 'NE',

 'New Hampshire': 'NH',

 'New Jersey': 'NJ',

 'New Mexico': 'NM',

 'Nevada': 'NV',

 'New York': 'NY',

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

 'Virginia': 'VA',

 'Virgin Islands': 'VI',

 'Vermont': 'VT',

 'Washington': 'WA',

 'Wisconsin': 'WI',

 'West Virginia': 'WV',

 'Wyoming': 'WY'}



statecodes = []

for i in statenames:

    statecodes.append(states[i])

import plotly.express as px



state_farms = pd.DataFrame(fm['State'].value_counts())

state_farms.head()

state_farms.rename(columns={'State':'Number of farmers markets'}, inplace = True)



px.bar(state_farms, x=state_farms.index, y='Number of farmers markets')
import plotly.graph_objects as go





fig = go.Figure(data=go.Scattergeo(

        lon = fm['x'],

        lat = fm['y'],

        mode = 'markers',

        text = fm['details'],

        marker = dict(size = 1,opacity = 1,reversescale = True,autocolorscale = False,symbol = 0, 

                      line = dict(width=1,color='rgba(102, 102, 102)'),colorscale = 'icefire',cmin = 0)

                )

        )

    



fig.update_layout(

        title = 'US farmers markets',

        geo_scope='usa'

    )

fig.show()
fig = px.choropleth(data_frame=av_state_per_capita_income,

                    locations=statecodes,

                    locationmode="USA-states",

                    color='per capita income',

                    color_continuous_scale = 'Blues',

                    scope="usa",

                    title = 'US States by average income per capita.'

                   )



fig.show()
fig = px.scatter(data_frame = markets,

           x = 'per capita income',

           y = 'no. farmers markets',

           size = 'population',

           color = 'population',

           color_continuous_scale = 'Blues',

           trendline = 'ols',

           trendline_color_override = 'red',

           hover_data = markets.columns,

           title = 'Relationship between per capita income and no. farmers markets.'

       )



fig.show()
fig = px.density_contour(markets,

                         x="per capita income",

                         y="no. farmers markets"

                        )



fig.update_traces(contours_coloring="fill",

                  contours_showlabels = True,

                 colorscale = 'Blues')



fig.show()
fig = px.scatter(data_frame = markets,

           x = 'median household income',

           y = 'no. farmers markets',

           trendline = 'ols',

           size='population',

           color = 'population',

           color_continuous_scale = 'emrld',

           hover_data = markets.columns,

           title = 'Relationship between median household income and no. farmers markets per state.'

          )



fig.show()

fig = px.density_contour(markets,

                         x="median household income",

                         y="no. farmers markets"

                        )



fig.update_traces(contours_coloring="fill",

                  contours_showlabels = True,

                 colorscale = 'Blues')



fig.show()
fig = px.choropleth(data_frame=markets,

                    locations=statecodes,

                    locationmode="USA-states",

                    color='population',

                    color_continuous_scale = 'greens',

                    scope="usa",

                    title = 'US States by population.'

                   )



fig.show()
fig = px.scatter(data_frame = markets,

           x = 'population',

           y = 'no. farmers markets',

           trendline = 'ols',

           size='median household income',

           color = 'median household income',

          color_continuous_scale = 'Blues',

           hover_data = markets.columns,

           title = 'Relationship between population and no. farmers markets per state.'

          )



fig.show()
fig = px.density_contour(markets,

                         x="population",

                         y="no. farmers markets"

                        )



fig.update_traces(contours_coloring="fill",

                  contours_showlabels = True,

                 colorscale = 'Blues')



fig.show()