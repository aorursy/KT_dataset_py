import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# the first question 
coutry_not_using_ppm = """
        SELECT DISTINCT country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit!='ppm'
        """
# excute above 
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(coutry_not_using_ppm)
pollutants_with_value0 = """SELECT DISTINCT pollutant
                   FROM `bigquery-public-data.openaq.global_air_quality`
                   WHERE value = 0
                   """
# excute above 
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(pollutants_with_value0)
#Save answer for question 1
coutry_not_using_ppm_csv('countries.csv', index=False)
#Save answer for question 2
pollutants_with_value0_csv('pollutants.csv', index=False)