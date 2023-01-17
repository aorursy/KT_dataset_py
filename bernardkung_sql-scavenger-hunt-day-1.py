import bq_helper as bqh
openaq = bqh.BigQueryHelper(active_project="bigquery-public-data", dataset_name= "openaq")
openaq.list_tables()
openaq.table_schema('global_air_quality')
openaq.head('global_air_quality')
query1 = """ 
SELECT country 
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
"""
openaq.estimate_query_size(query1)
nonppm_countries = openaq.query_to_pandas_safe(query1, max_gb_scanned = 0.001)
nonppm_countries.head()
nonppm_countries.shape
nonppm_countries_counttable = nonppm_countries.country.value_counts()
print(nonppm_countries_counttable)
query2 = """ 
SELECT pollutant, value
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""
openaq.estimate_query_size(query2)
zero_pollutants = openaq.query_to_pandas_safe(query2, max_gb_scanned=0.1)
zero_pollutants.shape
zero_pollutants.head()
zero_pollutants_table = zero_pollutants.pollutant.value_counts()
print(zero_pollutants_table)
query3 = """
SELECT DISTINCT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
"""

openaq.estimate_query_size(query3)
unique_pollutants = openaq.query_to_pandas_safe(query3, max_gb_scanned=0.001)
print(unique_pollutants)