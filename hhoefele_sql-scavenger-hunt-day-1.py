#import Kaggle's bq_helper package
import bq_helper
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper (active_project="bigquery-public-data",
                                    dataset_name = "openaq")
# print a list of all the tables in this dataset
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# print information on all the columns in the "global_air_quality" table
# in the open_aq dataset
open_aq.table_schema("global_air_quality")
# preview the first ten entries in the location column of the "global_air_quality" table
open_aq.head("global_air_quality", selected_columns="location", num_rows = 10)
# query to select all the items from the "city" column where the "country" column is "US"
# be sure to use backticks ` not single quotes '
query = """ SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
    """
#check how big this query will be
open_aq.estimate_query_size(query)
# run query
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# save dataframe as a .csv
us_cities.to_csv("air_quality_US_Cities_df.csv")
# query to select all the items from the "country" column where "pollutant" isn't ppm
query2 = """ SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit!= 'ppm'
    """
#check how big this query will be
open_aq.estimate_query_size(query2)
# create dataframe for query2 results
countries_not_ppm = open_aq.query_to_pandas_safe(query2)
# use dataframe to get answer to:
# Which countries use a unit other than ppm to measure any type of pollution?
countries_not_ppm.country.value_counts().head()
# save query dataframe to .csv 
countries_not_ppm.to_csv("countries_not_ppm.csv")
# Which pollutants have a value of exactly 0?
# query to select all the items from the "pollutant" column where "pollutant" equals zero
query3 = """ SELECT DISTINCT pollutant 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
    """
#check how big this query will be
open_aq.estimate_query_size(query3)
# create dataframe for query3 results
pollutant_zero = open_aq.query_to_pandas_safe(query3)

# use dataframe to get answer to:
# Which pollutants have a value of exactly 0?
pollutant_zero.pollutant.value_counts().head()

# save query dataframe to .csv 
pollutant_zero.to_csv("pollutant_zero.csv")