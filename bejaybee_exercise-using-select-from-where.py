# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# Your Code Goes Here
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
us_country = open_aq.query_to_pandas_safe(query)
us_country.country.value_counts()
# Your Code Goes Here
query = """SELECT location
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
us_pll = open_aq.query_to_pandas_safe(query)
us_pll.location.value_counts()
us_pll
