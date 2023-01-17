# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")
query = """
SELECT country, unit FROM `bigquery-public-data.openaq.global_air_quality`
"""
country_unit = open_aq.query_to_pandas_safe(query)
print(f"{country_unit.head()}")
country_unit.groupby(['country', 'unit']).size()
target_unit = [n for n in country_unit['unit'].unique() if n != 'ppm'][0]
target_unit
target_row = (country_unit['unit'] == target_unit)
country_unit.loc[target_row, 'country'].unique()
# Your Code Goes Here
open_aq.head("global_air_quality")
query = """SELECT pollutant, value
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""
poll_zero = open_aq.query_to_pandas_safe(query)
poll_zero.head()
print(f"{poll_zero['value'].nunique(), poll_zero['value'].unique()}")
print(f"{[n for n in poll_zero['pollutant'].unique()]}")