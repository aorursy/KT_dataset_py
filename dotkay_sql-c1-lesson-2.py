# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery
client = bigquery.Client()
dataset_ref = client.dataset('openaq', project='bigquery-public-data')

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))
for table in tables:

    print(table.table_id)
global_airquality_table_ref = dataset_ref.table('global_air_quality')

global_airquality_table = client.get_table(global_airquality_table_ref)
global_airquality_table.schema
client.list_rows(global_airquality_table, max_results=5).to_dataframe()
my_query= """

        SELECT city, country

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

"""

us_airqual = client.query(my_query)
us_airqual = client.query(my_query).to_dataframe()
us_airqual.city.value_counts().head()
us_airqual.city.unique()
india_query = """

        SELECT * 

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'IN'

"""

in_air_quality = client.query(india_query).to_dataframe()
in_air_quality.head()
in_air_quality.city.value_counts().head()
s_country_query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country LIKE 'S%'

"""

s_country_air_quality = client.query(s_country_query)
s_country_air_quality.to_dataframe().head()
num_pollutant_us_query = """

        SELECT pollutant, 

            REGEXP_CONTAINS(pollutant, r"[0-9]+") AS has_number

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

"""

client.query(num_pollutant_us_query).to_dataframe().head()