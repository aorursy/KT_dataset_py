import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import matplotlib.pyplot as plt

import folium

from shapely import wkt



import datetime as dt

import warnings



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

from plotly.subplots import make_subplots



warnings.filterwarnings(action="ignore")

np.set_printoptions(suppress=True)

pd.set_option("display.float_format", lambda x: "%.5f" % x)

pd.options.display.max_rows = 999



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_df = pd.read_csv('/kaggle/input/enrichednytimescovid19/covid19_us_county.csv')

print(covid_df.shape)
#turn to datetime from string

covid_df['date'] = pd.to_datetime(covid_df['date'])
# also want to merge in geo info for geospatial viz and analysis

geo_county_df = pd.read_csv('/kaggle/input/enrichednytimescovid19/us_county_pop_and_shps.csv')

geo_covid_df = pd.merge(covid_df, geo_county_df, how='left', on=['fips', 'county', 'state', 'county_pop_2019_est']).dropna()

geo_covid_df['date2'] = geo_covid_df['date'].astype('str')
covid_df.tail(2)
covid_df.describe()
# number of unique U.S. counties with at least 1 confirmed case?

#covid_df.fips.nunique()
# what is the date range?

print('begin: {0}, end: {1}'.format(covid_df.date.min(), covid_df.date.max()))
geo_covid_df['deaths_per_100k_pop'] = geo_covid_df['deaths_per_capita_100k']

fig = px.scatter_mapbox(geo_covid_df[geo_covid_df.date > dt.datetime(2020, 3, 14)], 

                        lat="county_center_lat", lon="county_center_lon", 

                        hover_name="county", hover_data=["state", "county"], color="deaths_per_100k_pop", 

                        size="deaths_per_capita_100k",size_max=30, zoom=3, height=500,

                        animation_frame=str('date2'))

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

py.offline.iplot(fig)
geo_covid_df['new cases per 100k pop.'] = geo_covid_df['new_day_cases_per_capita_100k']

fig = px.density_mapbox(geo_covid_df[geo_covid_df.date > dt.datetime(2020, 3, 14)], 

                        lat="county_center_lat", lon="county_center_lon", 

                        hover_name="county", hover_data=["state", "county", "new cases per 100k pop."], 

                        radius=10,

                        z='new cases per 100k pop.',

                        zoom=3, 

                        height=600,

                        width=1000,

                        title='new cases per 100k pop.',

                        animation_frame=str('date2'),

                       )

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

py.offline.iplot(fig)
us_df = covid_df.groupby(['date'], as_index=False).sum()



plt_titles = ['U.S. Total Cases', 'U.S. Total Deaths']

fig = make_subplots(rows=2, cols=1, subplot_titles=plt_titles)

fig.add_trace(go.Scatter(x=us_df['date'], y=us_df['cases']), row=1, col=1)

fig.add_trace(go.Scatter(x=us_df['date'], y=us_df['deaths']), row=2, col=1)

fig.update_xaxes(range=[dt.date(2020, 2, 1), covid_df.date.max() + dt.timedelta(days=2)])

fig.update_layout(title='U.S - COVID19 Cumulative Totals', height=700,

                  showlegend=False)
fig = px.bar(us_df, x='date', y='new_day_cases')

fig.update_layout(title = 'U.S -  New Cases per Day')

py.offline.iplot(fig)
# lets get just the latest date

latest_covid_df = covid_df[covid_df.date == covid_df.date.max()]

#latest_covid_df.head(2)

top10_cases_county_df = latest_covid_df.sort_values('cases_per_capita_100k', ascending=False).head(10)

top10_cases_county_df['state_county'] = top10_cases_county_df['state'] + '-' + top10_cases_county_df['county']



fig = go.Figure(data=[go.Bar(

            x=top10_cases_county_df['state_county'], y=top10_cases_county_df['cases_per_capita_100k'],

        )])

fig.update_layout(title='Top 10 Counties - Cases Per 100k People (as of {0})'.format(covid_df.date.max()))

py.offline.iplot(fig)
latest_covid_df['log_cases_per_capita_100k'] = latest_covid_df['cases_per_capita_100k'].apply(lambda x: np.log(x+1))

latest_covid_df['log_pop_per_sq_mile_2010'] = latest_covid_df['pop_per_sq_mile_2010'].apply(lambda x: np.log(x+1))

sns.jointplot(x="log_cases_per_capita_100k", y="log_pop_per_sq_mile_2010", kind="hex",

              data=latest_covid_df);
us_state_df = covid_df.groupby(['date', 'state'], as_index=False).sum()

us_state_df['date2'] = us_state_df['date'].astype('str')

max_date = covid_df.date.max()

fig = px.pie(us_state_df[us_state_df.date == max_date], 

             values='cases', names='state', title='U.S Total Cases Per State - {0}'.format(max_date))

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(height=700)

fig.show()
us_state_abbrev = {

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
us_state_df['state_code'] = us_state_df.apply(lambda x: us_state_abbrev.get(x.state,float('nan')), axis=1)

us_state_df['log(confirmedCases)'] = np.log(us_state_df['cases'] + 1)

us_state_df['log(deaths)'] = np.log(us_state_df['deaths'] + 1)
px.choropleth(us_state_df[us_state_df.date > dt.datetime(2020, 3, 1)],

              locationmode='USA-states',

              scope='usa',

              locations='state_code',

              color='log(confirmedCases)',

              hover_name='state',

              hover_data=["cases"],

              animation_frame=str('date2'),

              color_continuous_scale=px.colors.sequential.Darkmint,

              title = 'Total Cases growth for USA (Log. Scale)')
px.choropleth(us_state_df[us_state_df.date > dt.datetime(2020, 3, 1)],

              locationmode='USA-states',

              scope='usa',

              locations='state_code',

              color='log(deaths)',

              hover_name='state',

              hover_data=["deaths"],

              animation_frame=str('date2'),

              color_continuous_scale=px.colors.sequential.OrRd,

              title = 'Total deaths for USA (Log. Scale)')
fig = go.Figure()



states = latest_covid_df.groupby(['state'], as_index=False).sum().sort_values('deaths', ascending=False).head(50).state.values

for astate in reversed(states):

    astate_df = us_state_df[us_state_df['state'] == astate]

    fig.add_trace(go.Scatter(

        x=astate_df['date'], y=astate_df['deaths'],

        mode='lines',

        #line=dict(width=0.5, color='rgb(184, 247, 212)'),

        line=dict(width=0.5),

        stackgroup='one',

        #groupnorm='percent', # sets the normalization for the sum of the stackgroup,

        name=astate

    ))





fig.update_layout(

    title = 'States - Total Confirmed Deaths',

    showlegend=True,

    xaxis_type='date',

    xaxis=dict(

        range=(us_state_df.date.min(), us_state_df.date.max() + dt.timedelta(days=10))

    ),

    yaxis=dict(

        type='linear',

        title='Total Confirmed Deaths'

        #range=[1, 101],

        #ticksuffix='%'

    ))



fig.show()
state_df = latest_covid_df.groupby('state', as_index=False).sum()

state_df['cases_per_capita_100k'] = state_df['cases'] / state_df['county_pop_2019_est'] * 100000

state_df = state_df.sort_values('cases_per_capita_100k', ascending=False)



fig = go.Figure(data=[go.Bar(

            x=state_df['state'], y=state_df['cases_per_capita_100k'],

        )])

fig.update_layout(title='State - Cases Per 100k People (as of {0})'.format(covid_df.date.max()), width=1000)

py.offline.iplot(fig)
state_df = state_df.sort_values('cases', ascending=False)



fig = go.Figure(data=[go.Bar(

            x=state_df['state'], y=state_df['cases'],

        )])

fig.update_layout(title='State - Total Cases (as of {0})'.format(covid_df.date.max()), width=1000)

py.offline.iplot(fig)
georgia_df = covid_df[covid_df.state == 'Georgia'].sort_values('cases_per_capita_100k', ascending=False)

fig = px.line(georgia_df, x="date", y="cases_per_capita_100k", color='county')

fig.update_layout(title='Georgia - Total Cases per 100k people', height=700)

fig.show()
georgia_df = geo_covid_df[geo_covid_df['state'] == 'Georgia']

fig = px.scatter_mapbox(georgia_df[georgia_df.date > dt.datetime(2020, 3, 14)], 

                        lat="county_center_lat", lon="county_center_lon", 

                        hover_name="county", hover_data=["state", "county"], color="deaths_per_100k_pop", 

                        size="deaths_per_capita_100k",size_max=30, zoom=5, height=500,

                        animation_frame=str('date2'))

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

py.offline.iplot(fig)
ny_df = geo_covid_df[geo_covid_df['state'] == 'New York']

fig = px.scatter_mapbox(ny_df[ny_df.date > dt.datetime(2020, 3, 14)], 

                        lat="county_center_lat", lon="county_center_lon", 

                        hover_name="county", hover_data=["state", "county"], color="deaths_per_100k_pop", 

                        size="deaths_per_capita_100k",size_max=30, zoom=5, height=500,

                        animation_frame=str('date2'))

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

py.offline.iplot(fig)
ny_df = covid_df[covid_df.state == 'New York'].groupby(['date'], as_index=False).sum().sort_values('date')

rest_us_df = covid_df[covid_df.state != 'New York'].groupby(['date'], as_index=False).sum().sort_values('date')



ny_population = state_df[state_df['state'] == 'New York'].county_pop_2019_est.iloc[0]

rest_us_population = state_df[state_df['state'] != 'New York'].county_pop_2019_est.sum()



#calc percent too

ny_pop_percent = np.round((ny_population / (ny_population + rest_us_population)) * 100, 2)

rest_pop_percent = np.round((rest_us_population / (ny_population + rest_us_population)) * 100, 2)
fig = make_subplots(rows=1, cols=2, subplot_titles=['Total Confirmed Deaths', 'Population'])





fig.add_trace(go.Scatter(x=ny_df['date'], y=ny_df['deaths'],

   name='New York', marker_color='blue'

), row=1, col=1)



fig.add_trace(go.Scatter(x=rest_us_df['date'], y=rest_us_df['deaths'],

   name='Rest of U.S.', marker_color='crimson'

), row=1, col=1)





fig.update_yaxes(dtick=1000, title='Total Confirmed Deaths', row=1, col=1)

fig.update_xaxes(range=[dt.date(2020, 3, 1), covid_df.date.max() + dt.timedelta(days=1)], row=1, col=1)





fig.add_bar(

            x=['New York', 'rest of U.S.'], y=[ny_population, rest_us_population], 

            marker_color=['blue','red'],

            text=[str(ny_pop_percent)+ ' %', str(rest_pop_percent) + ' %'],

            textposition='auto',

            row=1, col=2

        )



fig.update_layout(height=500, showlegend=False)



py.offline.iplot(fig)
plt_titles = ['New York New Cases per Day', 'Rest of U.S. New Cases Per Day']

fig = make_subplots(rows=1, cols=2, subplot_titles=plt_titles)

fig.add_bar(x=ny_df['date'], y=ny_df['new_day_cases'], row=1, col=1)

fig.add_bar(x=rest_us_df['date'], y=rest_us_df['new_day_cases'], row=1, col=2)

#fig.update_xaxes(range=[dt.date(2020, 2, 1), covid_df.date.max() + dt.timedelta(days=2)])

fig.update_layout(height=700, showlegend=False)

py.offline.iplot(fig)
# merge in demographic info per county

demog_df = pd.read_csv('/kaggle/input/enrichednytimescovid19/us_county_demographics.csv')

demog_df.head(2)
covid_demo_df = pd.merge(covid_df, demog_df, on=['state_fips', 'county_fips'], how='left')

covid_demo_df = covid_demo_df[covid_demo_df.date == covid_demo_df.date.max()]
corr_cols = [

       'cases_per_capita_100k', 'deaths_per_capita_100k', 'pop_per_sq_mile_2010',

       'MALE_PERC', 'FEMALE_PERC', 'WHITE_POP_PERC',

       'BLACK_POP_PERC', 'ASIAN_POP_PERC', 'HISP_POP_PERC', 'AGE_OTO4',

       'AGE_5TO14', 'AGE_15TO24', 'AGE_25TO34', 'AGE_35TO44', 'AGE_45TO54',

       'AGE_55TO64', 'AGE_65TO74', 'AGE_75TO84', 'AGE_84PLUS']

corr = covid_demo_df[corr_cols].corr()

plt.figure(figsize=(10,6))

sns.heatmap(corr)
merge_df = pd.merge(latest_covid_df, geo_county_df, how='left', on=['fips', 'county', 'state', 'county_pop_2019_est']).dropna()

#print(merge_df.shape)
merge_df['geom'] =  merge_df['county_geom'].apply(wkt.loads)

mdf = gpd.GeoDataFrame(merge_df, geometry='geom')

mdf.crs = "EPSG:4326"
variable = 'cases_per_capita_100k'

# set the range for the choropleth

vmin, vmax = 1, 100

# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(30, 10))

# create map

mdf[['geom', variable]].plot(variable, cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8')



sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm._A = []

# add the colorbar to the figure

cbar = fig.colorbar(sm)
mar31_df = covid_df[covid_df.date == '2020-03-31']

#mar31_df.head(2)
merge_df = pd.merge(mar31_df, geo_county_df, how='left', on=['fips', 'county', 'state', 'county_pop_2019_est']).dropna()

merge_df['geom'] =  merge_df['county_geom'].apply(wkt.loads)

mdf = gpd.GeoDataFrame(merge_df, geometry='geom')

mdf.crs = "EPSG:4326"

#print(mdf.shape)
col = 'cases_per_capita_100k'



mdf2 = mdf[['geom', 'county', col]]



bins = list(mdf2[col].quantile([0, 0.25, 0.5, 0.75, 1]))



m = folium.Map(location=[48, -102], zoom_start=3)



folium.Choropleth(

    geo_data=mdf2,

    name='choropleth',

    data=mdf2,

    columns=['county', col],

    key_on='feature.properties.county',

    fill_color='Reds',

    fill_opacity=0.7,

    line_opacity=0.2,

    legend_name='March 31st - County new day cases per 100,000 people',

    bins=bins,

    reset=True

).add_to(m)



folium.LayerControl().add_to(m)



m