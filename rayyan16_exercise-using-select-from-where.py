# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# Your Code Goes Here
open_aq.head("global_air_quality")
query = """ SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
        """
countries_not_ppm = open_aq.query_to_pandas_safe(query)
countries_not_ppm.country.value_counts().head()
# Your Code Goes Here
query1 = """ SELECT pollutant
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE value = 0.00
         """
zero_pollutant = open_aq.query_to_pandas_safe(query1)
zero_pollutant.pollutant.value_counts()
