# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head('global_air_quality')
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
        """
countris_not_ppm = open_aq.query_to_pandas_safe(query)
# from now on countris_not_ppm is a dataframe which is nice :)
countris_not_ppm.head()
# countris_not_ppm.country.value_counts()
countris_not_ppm.country.unique()
open_aq.head('global_air_quality')
query2 = """SELECT location
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """
zero_pollutant = open_aq.query_to_pandas_safe(query2)
zero_pollutant.head(10)
# zero_pollutant.location.unique()