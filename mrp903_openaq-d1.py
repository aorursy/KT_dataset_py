import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
openaq_obj = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='openaq')
openaq_obj.list_tables()
openaq_obj.table_schema('global_air_quality')
openaq_obj.head('global_air_quality')
## First query (just to test things!):
query_0 = """ SELECT location, city, country 
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE country = 'US' """
print (openaq_obj.estimate_query_size(query_0)*1024, 'MB')
query_1 = """ SELECT DISTINCT country, unit 
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE unit != 'ppm' 
              ORDER BY country"""

print (openaq_obj.estimate_query_size(query_1)*1024, 'MB')
no_ppm_unit = openaq_obj.query_to_pandas(query = query_1)

print ('Following coutries use unit other than ppm:')
no_ppm_unit
## Question-2: which pollutants have value = 0?
query_2 = """ SELECT DISTINCT pollutant, value
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE value = 0 """
print (openaq_obj.estimate_query_size(query_2)*1024, 'MB')
pollutants_zero_val = openaq_obj.query_to_pandas(query = query_2)

print ('Following pollutants have value = 0:')
pollutants_zero_val['pollutant']
