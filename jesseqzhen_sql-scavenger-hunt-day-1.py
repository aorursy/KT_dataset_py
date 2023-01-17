# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
open_aq.estimate_query_size(query)
country_unit_not_ppm = open_aq.query_to_pandas_safe(query)
country_unit_not_ppm.country.unique()
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
open_aq.estimate_query_size(query)
pollutant_value_0 = open_aq.query_to_pandas_safe(query)
pollutant_value_0.pollutant.unique()
print("The pollutants that have a value of exactly 0 are:")
for x in pollutant_value_0.pollutant.unique():
    print(x)
