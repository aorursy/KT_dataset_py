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
query=""" SELECT country 
          FROM `bigquery-public-data.openaq.global_air_quality`
          WHERE unit != 'ppm'


"""


country_without_ppm = open_aq.query_to_pandas_safe(query)
#List of countries uses a unit other than ppm to measure pollution
country_without_ppm.country.unique().to_list()
query="""SELECT DISTINCT pollutant 
         FROM `bigquery-public-data.openaq.global_air_quality`
         WHERE value = 0
         """
pollutants_Zero_value = open_aq.query_to_pandas_safe(query)
pollutants_Zero_value
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
query=""" SELECT type, COUNT(id)
          FROM `bigquery-public-data.hacker_news.full`
          GROUP BY type
          """
          
stories_type = hacker_news.query_to_pandas_safe(query)
stories_type.head()
# Query for counting deleted comments
query2 = """SELECT COUNTIF(deleted=True)
            FROM `bigquery-public-data.hacker_news.comments`
            """
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments.head()
