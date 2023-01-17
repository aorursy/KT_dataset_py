import numpy as np # linear algebra

import pandas as pd

# https://github.com/SohierDane/BigQuery_Helper

from bq_helper import BigQueryHelper



bq_assistant = BigQueryHelper(active_project= "bigquery-public-data", dataset_name= "noaa_gsod")
bq_assistant.list_tables()

QUERY = """

        SELECT year, AVG(data.temp) AS avg_temp, AVG(data.max) AS avg_max, AVG(data.min) AS avg_min

        FROM `bigquery-public-data.noaa_gsod.gsod1929` AS data

        GROUP BY year

        """



bq_assistant.estimate_query_size(QUERY)
df = bq_assistant.query_to_pandas_safe(QUERY)

df.head()
START_YEAR = 1930

END_YEAR = 2019



for year in range(START_YEAR, END_YEAR):

   QUERY = """

        SELECT year, AVG(data.temp) AS avg_temp, AVG(data.max) AS avg_max, AVG(data.min) AS avg_min

        FROM `bigquery-public-data.noaa_gsod.gsod{}` AS data

        GROUP BY year

        """.format(year)

   df_temp = bq_assistant.query_to_pandas_safe(QUERY)

   df = df.append(df_temp, ignore_index=True)

   print ("Added {}".format(year))

df.head()
df.to_csv("avg_temps_1929_2018.csv", index=False)

df.to_csv("avg_temps_1929_2018_indexed.csv")

df.to_json("avg_temps_1929_2018_indexed.json")