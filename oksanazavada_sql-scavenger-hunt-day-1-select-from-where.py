# import our bq_helper package
import bq_helper 

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")

# print information on all the columns in the "full" table
# in the open_aq dataset
open_aq.table_schema("global_air_quality")

# query to select countries which use a unit other than ppm to measure any type of pollution
# columns "country" and "unit" != ppm
query_not_ppm_unit = """SELECT DISTINCT country
                        FROM `bigquery-public-data.openaq.global_air_quality`
                        WHERE  unit != 'ppm' and unit is not NULL
                        """

# check how big this query will be
open_aq.estimate_query_size(query_not_ppm_unit)

# only run this query if it's less than 1 gigabyte by default
job_1 = open_aq.query_to_pandas_safe(query_not_ppm_unit)

# save our dataframe as a .csv 
job_1.to_csv("SQL_Scavenger_Hunt_task1.csv")

# query to select pollutants which have a value of exactly 0
# columns "pollutant" and "value" = 0.00
query_pollutant_value_0 = """SELECT DISTINCT pollutant
                        FROM `bigquery-public-data.openaq.global_air_quality`
                        WHERE  value = 0.00 and value is not NULL
                        """

# check how big this query will be
open_aq.estimate_query_size(query_pollutant_value_0)

# only run this query if it's less than 1 gigabyte by default
job_2 = open_aq.query_to_pandas_safe(query_pollutant_value_0)

# save our dataframe as a .csv 
job_2.to_csv("SQL_Scavenger_Hunt_task2.csv")