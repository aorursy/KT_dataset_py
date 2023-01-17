import pandas as pd
from bq_helper import BigQueryHelper as BQH

open_aq = BQH(active_project="bigquery-public-data", 
              dataset_name="openaq")
open_aq.list_tables()


schema = open_aq.table_schema('global_air_quality')
properties = ('name', 'field_type', 'is_nullable', 'description')
schema_df = pd.DataFrame([[eval('field.' + p) for field in schema] 
              for p in properties]).transpose()
schema_df.columns = properties
schema_df
#Which countries use a unit other than ppm to measure any type of pollution?

ppm_query = """
    SELECT country AS code, unit
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit != 'ppm'
    GROUP BY country, unit
    ORDER BY country ASC"""

ppm_results = open_aq.query_to_pandas_safe(ppm_query)

country_names = pd.read_csv('../input/wikipedia-iso-country-codes.csv')
country_names.columns = ['country', '2code', '3code', 'numcode', 'iso']

ppm_results = pd.merge(ppm_results, country_names, 
                       left_on = 'code', right_on = '2code', 
                       how = 'left')
ppm_results = ppm_results[['code', 'country', 'unit']]
ppm_results
#Which pollutants have a value of exactly 0?
pollutants_query = """
SELECT pollutant, COUNT(country) AS n_measurements
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
GROUP BY pollutant
ORDER BY n_measurements DESC
"""

pollutants_df = open_aq.query_to_pandas_safe(pollutants_query)
pollutants_df