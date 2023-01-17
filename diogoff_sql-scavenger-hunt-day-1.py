import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                   dataset_name='openaq')
open_aq.list_tables()
open_aq.head('global_air_quality')
query = '''SELECT DISTINCT unit
           FROM `bigquery-public-data.openaq.global_air_quality`
           ORDER BY unit
           '''
open_aq.estimate_query_size(query)
units = open_aq.query_to_pandas_safe(query)
units
query = '''SELECT DISTINCT country
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit <> 'ppm'
           ORDER BY country
           '''
open_aq.estimate_query_size(query)
countries_not_ppm = open_aq.query_to_pandas_safe(query)
countries_not_ppm
query = '''SELECT unit, COUNT(DISTINCT country) AS num_countries
           FROM `bigquery-public-data.openaq.global_air_quality`
           GROUP BY unit
           '''
open_aq.estimate_query_size(query)
num_countries_by_unit = open_aq.query_to_pandas_safe(query)
num_countries_by_unit
num_countries_by_unit.plot(x='unit', y='num_countries', kind='bar')
query = '''SELECT DISTINCT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
           ORDER BY pollutant
           '''
open_aq.estimate_query_size(query)
pollutants_with_zero = open_aq.query_to_pandas_safe(query)
pollutants_with_zero
query = '''SELECT pollutant, COUNT(*) AS num_zeros
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
           GROUP BY pollutant
           ORDER BY num_zeros DESC
           '''
open_aq.estimate_query_size(query)
num_zeros_by_pollutant = open_aq.query_to_pandas_safe(query)
num_zeros_by_pollutant
num_zeros_by_pollutant.plot(x='pollutant', y='num_zeros', kind='bar')
