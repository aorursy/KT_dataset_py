# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset
open_aq.list_tables()

#Query that returns all the country names which do not use the units of ppm
query = """SELECT DISTINCT country 
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        """

#Check whether the qery is safe to run
other_than_ppm = open_aq.query_to_pandas_safe(query)

# Display the Query result
other_than_ppm
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")

query = """SELECT DISTINCT pollutant 
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """
# Check if its safe to run the query
zero_value_pollutant = open_aq.query_to_pandas_safe(query)

#Display the Query results
zero_value_pollutant
