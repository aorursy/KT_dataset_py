# pandas for handling data
import pandas as pd
# google bigquery library for quering data
from google.cloud import bigquery
# BigQueryHelper for converting query result direct to dataframe
from bq_helper import BigQueryHelper
# matplotlib for plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# import plotly
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

%matplotlib inline
QUERY = """
    SELECT
        co_daily.state_name,
        avg(co_daily.aqi) as co_avg_aqi,
        avg(no_daily.aqi) as no_avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.co_daily_summary` AS co_daily
    INNER JOIN `bigquery-public-data.epa_historical_air_quality.no2_daily_summary` AS no_daily
        ON co_daily.state_name = no_daily.state_name
    WHERE
      co_daily.poc = 1
      AND no_daily.poc = 1
      AND EXTRACT(YEAR FROM co_daily.date_local) = 2015
      AND EXTRACT(YEAR FROM no_daily.date_local) = 2015
    GROUP BY co_daily.state_name
        """

bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
df_states_gas = bq_assistant.query_to_pandas(QUERY)
QUERY = """
    SELECT
        o3_daily.state_name,
        avg(o3_daily.aqi) as o3_avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.o3_daily_summary` AS o3_daily
    WHERE
      o3_daily.poc = 1
      AND EXTRACT(YEAR FROM o3_daily.date_local) = 2015
    GROUP BY o3_daily.state_name
        """
df_states_gas_o3 = bq_assistant.query_to_pandas(QUERY)
QUERY = """
    SELECT
        so2_daily.state_name,
        avg(so2_daily.aqi) as so2_avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.so2_daily_summary` AS so2_daily
    WHERE
      so2_daily.poc = 1
      AND EXTRACT(YEAR FROM so2_daily.date_local) = 2015
    GROUP BY so2_daily.state_name
        """
df_states_gas_so2 = bq_assistant.query_to_pandas(QUERY)
df_states_gas['o3_avg_aqi'] = df_states_gas['state_name'].map(df_states_gas_o3.set_index('state_name')['o3_avg_aqi'])
df_states_gas['so2_avg_aqi'] = df_states_gas['state_name'].map(df_states_gas_so2.set_index('state_name')['so2_avg_aqi'])
states = {'AL': 'Alabama',
'AK': 'Alaska',
'AZ':'Arizona',
'AR':'Arkansas',
'CA':'California',
'CO':'Colorado',
'CT':'Connecticut',
'DE':'Delaware',
'FL':'Florida',
'GA':'Georgia',
'HI':'Hawaii',
'ID':'Idaho',
'IL':'Illinois',
'IN':'Indiana',
'IA':'Iowa',
'KS':'Kansas',
'KY':'Kentucky',
'LA':'Louisiana',
'ME':'Maine',
'MD':'Maryland',
'MA':'Massachusetts',
'MI':'Michigan',
'MN':'Minnesota',
'MS':'Mississippi',
'MO':'Missouri',
'MT':'Montana',
'NE':'Nebraska',
'NV':'Nevada',
'NH':'New Hampshire',
'NJ':'New Jersey',
'NM':'New Mexico',
'NY':'New York',
'NC':'North Carolina',
'ND':'North Dakota',
'OH':'Ohio',
'OK':'Oklahoma',
'OR':'Oregon',
'PA':'Pennsylvania',
'RI':'Rhode Island',
'SC':'South Carolina',
'SD':'South Dakota',
'TN':'Tennessee',
'TX':'Texas',
'UT':'Utah',
'VT':'Vermont',
'VA':'Virginia',
'WA':'Washington',
'WV':'West Virginia',
'WI':'Wisconsin',
'WY':'Wyoming'}
df_states = pd.DataFrame.from_dict(states,orient='index').reset_index()
df_states.columns = ['code', 'code_name']
df_states_gas['state_code'] = df_states_gas['state_name'].map(df_states.set_index('code_name')['code'])
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],[0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_states_gas['state_code'],
        z = df_states_gas['o3_avg_aqi'].astype(float),
        locationmode = 'USA-states',
        text =  'Average AQI of NO: ' + df_states_gas['no_avg_aqi'].astype(str) + '<br>' + 'Average AQI of CO: ' + df_states_gas['co_avg_aqi'].astype(str) + '<br>' + 'Average AQI of SO2: ' + df_states_gas['so2_avg_aqi'].astype(str),
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "AQI of O3")
        ) ]

layout = dict(
        title = 'The average air quality index of some dangerous element in different US states<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )




    
fig = dict( data=data, layout=layout )

py.iplot( fig, filename='d3-cloropleth-map' )
# the data is stored in Google big query. In order to get the data the only way possible is using SQL
# we will need date, state name and aqi column from database table and we add filter where poc will only be 1
# and for simplicity purpose I take some random date

QUERY = """
    SELECT
        EXTRACT(YEAR FROM date_local) as day_of_year,
        avg(aqi) as avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
    WHERE
      poc = 1
    GROUP BY day_of_year
    ORDER BY day_of_year ASC
        """

bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
df_co = bq_assistant.query_to_pandas_safe(QUERY)
# df_co = df_co.dropna()
# df_co.day_of_year = pd.to_datetime(df_co.day_of_year)
plt.subplots(figsize=(15,7))
sns.barplot(x='day_of_year',y='avg_aqi',data=df_co,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Air Quality Index', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Year', fontsize=20)
plt.title('Average AQI of Carbon monoxide in different years', fontsize=24)
plt.show()
QUERY = """
    SELECT
        EXTRACT(YEAR FROM date_local) as day_of_year,
        MAX(aqi) as avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
    WHERE
      poc = 1
    GROUP BY day_of_year
    ORDER BY day_of_year ASC
        """

df_co_max = bq_assistant.query_to_pandas_safe(QUERY)
plt.subplots(figsize=(15,7))
sns.barplot(x='day_of_year',y='avg_aqi',data=df_co_max,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Air Quality Index', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Year', fontsize=20)
plt.title('Highest Average AQI of Carbon monoxide in different years', fontsize=24)
plt.savefig('high_carbon.png')
plt.show()
# the data is stored in Google big query. In order to get the data the only way possible is using SQL
# we will need date, state name and aqi column from database table and we add filter where poc will only be 1
# and for simplicity purpose I take some random date

QUERY = """
    SELECT
        state_name,
        avg(aqi) as avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
    WHERE
      poc = 1
    GROUP BY state_name
    ORDER BY avg_aqi ASC
        """

# bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
df_co_states = bq_assistant.query_to_pandas_safe(QUERY)
plt.subplots(figsize=(15,7))
sns.barplot(x='state_name',y='avg_aqi',data=df_co_states,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Air Quality Index', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Year', fontsize=20)
plt.title('Average AQI of Carbon monoxide in different states', fontsize=24)
plt.show()
QUERY = """
    SELECT
        EXTRACT(YEAR FROM date_local) as day_of_year,
        avg(aqi) as avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.o3_daily_summary`
    WHERE
      poc = 1
    GROUP BY day_of_year
    ORDER BY day_of_year ASC
        """
df_ozone = bq_assistant.query_to_pandas_safe(QUERY)
plt.subplots(figsize=(15,7))
sns.barplot(x='day_of_year',y='avg_aqi',data=df_ozone,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Air Quality Index of Ozone gas', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Year', fontsize=20)
plt.title('Mean AQI of Ozone gas in different years', fontsize=24)
plt.show()
QUERY = """
    SELECT
        EXTRACT(YEAR FROM date_local) as day_of_year,
        MAX(aqi) as avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.o3_daily_summary`
    WHERE
      poc = 1
    GROUP BY day_of_year
    ORDER BY day_of_year ASC
        """
df_ozone_max = bq_assistant.query_to_pandas_safe(QUERY)
plt.subplots(figsize=(15,7))
sns.barplot(x='day_of_year',y='avg_aqi',data=df_ozone_max,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Air Quality Index of Ozone gas', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Year', fontsize=20)
plt.title('Highest AQI of Ozone gas in different years', fontsize=24)
plt.show()
QUERY = """
    SELECT
        state_name,
        avg(aqi) as avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.o3_daily_summary`
    WHERE
      poc = 1
    GROUP BY state_name
    ORDER BY avg_aqi ASC
        """

df_ozone_states = bq_assistant.query_to_pandas_safe(QUERY)
plt.subplots(figsize=(15,7))
sns.barplot(x='state_name',y='avg_aqi',data=df_ozone_states,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Air Quality Index', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('States', fontsize=20)
plt.title('Mean AQI of Ozone gas in different states', fontsize=24)
plt.show()
QUERY = """
    SELECT
        EXTRACT(YEAR FROM date_local) as day_of_year,
        avg(aqi) as avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.so2_daily_summary`
    WHERE
      poc = 1
    GROUP BY day_of_year
    ORDER BY day_of_year ASC
        """

df_sulfur = bq_assistant.query_to_pandas_safe(QUERY)
plt.subplots(figsize=(15,7))
sns.barplot(x='day_of_year',y='avg_aqi',data=df_sulfur,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Air Quality Index of SO2', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Year', fontsize=20)
plt.title('Mean of AQI of SO2 gas in different years', fontsize=24)
plt.show()
QUERY = """
    SELECT
        state_name,
        avg(aqi) as avg_aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.so2_daily_summary`
    WHERE
      poc = 1
    GROUP BY state_name
    ORDER BY avg_aqi ASC
        """
df_sulfur_states = bq_assistant.query_to_pandas_safe(QUERY)
plt.subplots(figsize=(15,7))
sns.barplot(x='state_name',y='avg_aqi',data=df_ozone_states,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Air Quality Index', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('States', fontsize=20)
plt.title('Mean AQI of Ozone gas in different states', fontsize=24)
plt.show()
































