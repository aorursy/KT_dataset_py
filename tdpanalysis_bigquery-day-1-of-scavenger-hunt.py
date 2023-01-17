import bq_helper as bq
open_aq = bq.BigQueryHelper(active_project="bigquery-public-data",
                           dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
query = """SELECT DISTINCT(country),
                  unit
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE LOWER(unit) != "ppm" 
           ORDER BY country ASC
"""
not_ppm_countries = open_aq.query_to_pandas(query)
not_ppm_countries.head()
len(not_ppm_countries.country)
print(not_ppm_countries.country.tolist())
query2 = """SELECT DISTINCT(pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant ASC
"""
zero_pollutants = open_aq.query_to_pandas_safe(query2)
zero_pollutants.head(10)