import pandas as pd
import bq_helper
# This tells to create an object which uses an active project called bigquery-public-data (seems to be)
# default for all kernels using BigQuery on Kaggle, and the dataset name is openaq (~DB Name)
open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                  dataset_name = "openaq")
# Listing all tables in the db
print(open_aq.list_tables())

# Showing 5 lines of the table. The syntax is different from the pandas head
open_aq.head("global_air_quality")
# Defining the query and getting country name only
query = """
SELECT
    DISTINCT country
FROM `bigquery-public-data.openaq.global_air_quality` # ProjectName.DSN.TableName
WHERE unit != 'ppm'
"""
# Running the query only if scan size is less than 1GB
country_non_ppm = open_aq.query_to_pandas_safe(query)
print(str(country_non_ppm.shape[0]) + " countries measure pollutants in units other than ppm. Some of them are:")
country_non_ppm.head(n=5)
query2 = """
SELECT 
    DISTINCT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""

zero_pollution = open_aq.query_to_pandas_safe(query2)
print(str(zero_pollution.shape[0])+ " pollutants have 0 values on certain days and locations. They are:" )
zero_pollution