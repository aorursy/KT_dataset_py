# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()


#the two queries
no_ppm_query = """SELECT country
                  FROM `bigquery-public-data.openaq.global_air_quality`
                  WHERE unit != 'ppm'
               """

zero_pollutant_query = """SELECT pollutant
                          FROM `bigquery-public-data.openaq.global_air_quality`
                          WHERE value = 0.
                       """

#queries converted to safe dataframes
no_ppm_df = open_aq.query_to_pandas_safe(no_ppm_query)
zero_pollutant_df = open_aq.query_to_pandas_safe(zero_pollutant_query)

#results
print("The following countries do not use ppm as a unit:")
print(no_ppm_df.country.unique())
print()
print("The following pollutants have a value of zero:")
print(zero_pollutant_df.pollutant.unique())
