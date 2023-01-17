import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.express as px

from google.cloud import bigquery

client = bigquery.Client()

dataset = client.get_dataset('bigquery-public-data.covid19_google_mobility')

tables = list(client.list_tables(dataset))
print('New York City')

sql = '''

SELECT

  *

FROM

  `bigquery-public-data.covid19_google_mobility.mobility_report` 

WHERE

  country_region = "United States"

  AND sub_region_1 = "New York"

  AND sub_region_2 = "New York County"

  AND date BETWEEN "2020-01-10" AND "2020-09-07"

ORDER BY

  date

'''

query_job = client.query(sql)

df = query_job.to_dataframe()

fig = plt.figure();

df.plot(x='date', rot=45, y=['retail_and_recreation_percent_change_from_baseline',                                  

                             'grocery_and_pharmacy_percent_change_from_baseline',

                             'parks_percent_change_from_baseline',

                             'transit_stations_percent_change_from_baseline',

                             'workplaces_percent_change_from_baseline',

                             'residential_percent_change_from_baseline'])

plt.legend(bbox_to_anchor=(1, 0.5), loc='lower left')

plt.xlabel('Date')

plt.ylabel('Percent Change From Baseline')

plt.show()
print('San Francisco')

sql = '''

SELECT

  *

FROM

  `bigquery-public-data.covid19_google_mobility.mobility_report` 

WHERE

  country_region = "United States"

  AND sub_region_1 = "California"

  AND sub_region_2 = "San Francisco County"

  AND date BETWEEN "2020-01-10" AND "2020-09-07"

ORDER BY

  date

'''

query_job = client.query(sql)

df = query_job.to_dataframe()

fig = plt.figure();

df.plot(x='date', rot=45, y=['retail_and_recreation_percent_change_from_baseline',                                  

                             'grocery_and_pharmacy_percent_change_from_baseline',

                             'parks_percent_change_from_baseline',

                             'transit_stations_percent_change_from_baseline',

                             'workplaces_percent_change_from_baseline',

                             'residential_percent_change_from_baseline'])

plt.legend(bbox_to_anchor=(1, 0.5), loc='lower left')

plt.xlabel('Date')

plt.ylabel('Percent Change From Baseline')

plt.show()