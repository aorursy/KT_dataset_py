# import pandas
import pandas as pd

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# Your code goes here :)

# Query 1 selects the countries (and units, in case we want to examine that in the dataframe as well) 
# where the unit of measurement is not ppm
query1 = """
        SELECT DISTINCT country, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE NOT unit='ppm' 
"""

nonppm_countries = open_aq.query_to_pandas_safe(query1)
# Then we print out the unique values as a list of the countries (by 2-character Country Code)
print(nonppm_countries.country.unique())
# Query 2 selects those pollutants having a value of exactly zero

query2 = """
        SELECT pollutant, value
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value=0
        """

low_val_pols = open_aq.query_to_pandas_safe(query2)
print(low_val_pols.pollutant.unique())
for pollutant in low_val_pols.pollutant.unique():
    print(low_val_pols[low_val_pols['pollutant'] == pollutant].head(1))
