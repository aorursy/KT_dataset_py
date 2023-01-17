# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
#Selecting only countries that don't use ppm for measurement of pollution
query = """ SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

non_ppm_countries = open_aq.query_to_pandas_safe(query)

non_ppm_countries.head()
pollute_query = """ SELECT pollutant
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE value = 0.00
                """

zero_pollutants = open_aq.query_to_pandas_safe(pollute_query)

zero_pollutants.head()