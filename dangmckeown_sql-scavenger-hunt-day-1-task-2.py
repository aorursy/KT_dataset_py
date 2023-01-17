#SQL scavenger hunt day 1 task 2

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

#open_aq.head("global_air_quality")

# query task: Which pollutants have a value of zero?

query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            GROUP BY pollutant"""



no_value_pollutant = open_aq.query_to_pandas_safe(query)


no_value_pollutant.pollutant
