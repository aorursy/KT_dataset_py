# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)

# get the schema
open_aq.table_schema("global_air_quality")

unit_query = """ SELECT country 
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != 'ppm'"""

# check the query size
open_aq.estimate_query_size(unit_query)

# query size is low, so load data into a dataframe
countries = open_aq.query_to_pandas(unit_query)

# value_counts is a shortcut to group by and count by country (in this case). 
# shape gives the dimensions of the result, so we know there are 64 countries that don't use ppm for something
# (or rather, do use not ppm for something)
countries['country'].value_counts("country").shape

countrylist = countries['country'].drop_duplicates()
#countrylist
for i in range(0,countrylist.shape[0]):
    print('{i} {c}'.format(i=i,c=countrylist.iloc[i]))
zero_query = """SELECT pollutant
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value = 0 """

# check the query size
open_aq.estimate_query_size(zero_query)
# query size is low, so load data into a dataframe
pollutants = open_aq.query_to_pandas(zero_query)
pollutants['pollutant'].value_counts("pollutant")
