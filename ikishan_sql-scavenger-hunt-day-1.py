#Imporing Helper Package
import bq_helper
#Helper object for our bigguery dataset
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
# print a list of all the tables in the hacker_news dataset
open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            From `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
            """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()

#slecting countries where units are not PPM
query2 = """SELECT country
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != 'PPM'
                """
ppm_query = open_aq.query_to_pandas_safe(query2)
# What five most countries dosent have PPM as a unit?
ppm_query.country.value_counts().head()
query3 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            """
value_query = open_aq.query_to_pandas_safe(query3)
#pollutants with 0 value
value_query.pollutant.value_counts()
