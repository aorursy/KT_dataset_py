import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

from google.cloud import bigquery
client = bigquery.Client()



# List the tables in covid19_usafacts dataset which resides in bigquery-public-data project:

dataset = client.get_dataset('bigquery-public-data.covid19_usafacts')

tables = list(client.list_tables(dataset))

print([table.table_id for table in tables])
sql = '''

SELECT

  covid19.state,

  ROUND(sum(confirmed_cases/total_pop *100000), 2) AS confirmed_cases_per_100000,

  ROUND(sum(deaths/total_pop *100000), 2) AS deaths_per_100000

FROM `bigquery-public-data.covid19_usafacts.summary` covid19

JOIN `bigquery-public-data.census_bureau_acs.county_2017_5yr` acs 

ON covid19.county_fips_code = acs.geo_id

WHERE date = DATE_SUB(CURRENT_DATE(), INTERVAL 3 day) # yesterday

AND county_fips_code != "00000"

AND confirmed_cases + deaths > 0

GROUP BY covid19.state

ORDER BY confirmed_cases_per_100000 DESC, deaths_per_100000 DESC

'''

# Set up the query

query_job = client.query(sql)



# Make an API request  to run the query and return a pandas DataFrame

df = query_job.to_dataframe()

df.head(10)
fig = px.choropleth(df, locationmode="USA-states", locations='state', color='confirmed_cases_per_100000', scope="usa")

fig.show()
fig = px.choropleth(df, locationmode="USA-states", locations='state', color='deaths_per_100000', scope="usa")

fig.show()
sql = '''

SELECT

  covid19.state,

  total_pop AS state_population,

  confirmed_cases,

  ROUND(confirmed_cases/total_pop *100000,2) AS confirmed_cases_per_100000,

  deaths, 

  ROUND(deaths/total_pop *100000,2) AS deaths_per_100000,

FROM

  `bigquery-public-data.covid19_usafacts.summary` covid19

JOIN

  `bigquery-public-data.census_bureau_acs.state_2017_5yr` acs ON covid19.state_fips_code = acs.geo_id

WHERE

  date = DATE_SUB(CURRENT_DATE(),INTERVAL 1 day) 

  AND county_fips_code = "00000"

ORDER BY

  ROUND(confirmed_cases/total_pop *100000,2) desc,

  ROUND(deaths/total_pop *100000,2) desc

'''

# Set up the query

query_job = client.query(sql)



# Make an API request  to run the query and return a pandas DataFrame

df = query_job.to_dataframe()

df.head(10)