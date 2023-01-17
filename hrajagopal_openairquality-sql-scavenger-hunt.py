import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bq_helper
openAQ = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
openAQ.list_tables()
openAQ.table_schema("global_air_quality")

query_not_ppm = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != "PPM" """
openAQ.estimate_query_size(query_not_ppm)
openAQ_not_ppm = openAQ.query_to_pandas_safe(query_not_ppm)
openAQ_not_ppm.country.value_counts().head()

#query_not_ppm_distinct_countries = """SELECT distinct(country)
     #       FROM `bigquery-public-data.openaq.global_air_quality`
   #         WHERE pollutant != "PPM" """
#openAQ_not_ppm_distinct_countries = openAQ.query_to_pandas_safe(query_not_ppm_distinct_countries)




import bq_helper
openAQ = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
openAQ.list_tables()
openAQ.table_schema("global_air_quality")
query_zero_value_pollutants = """SELECT distinct(pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0 """
query_zero_value_pollutants = openAQ.query_to_pandas_safe(query_zero_value_pollutants)
query_zero_value_pollutants.pollutant.value_counts()


