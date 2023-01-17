



# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")







#query to select countries which use unit other than 'ppm'
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit)!='ppm'"""

#query will execute only if its less than 1GB
no_ppm_countries = open_aq.query_to_pandas_safe(query)

#display the results
no_ppm_countries


#query to select pollutants which has exactly value equal to zero
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0"""

zero_pollutants = open_aq.query_to_pandas_safe(query)

#Displaying the results
zero_pollutants
