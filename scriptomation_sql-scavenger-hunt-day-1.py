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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
#open_aq.list_tables()
#open_aq.head('global_air_quality')

#Query to find all countries that do not use ppm as the unit
pollutants_query = """SELECT country, unit
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE unit != 'ppm'
                    """
#Dataframe for query
other_pollutants = open_aq.query_to_pandas_safe(pollutants_query)

#dataframe shows all countries
other_pollutants.country.all()

#query to look for all pollutants with a value of 0
pollutants_none_query = """SELECT pollutant
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE value = 0
                    """

#creates a dataset based on the previous query
zero_pollutants = open_aq.query_to_pandas_safe(pollutants_none_query)

#displays all polutants with a value of exactly 0 
zero_pollutants.pollutant.all()
