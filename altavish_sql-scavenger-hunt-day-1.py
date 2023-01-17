# import package with helper functions 
import bq_helper as bq

# create a helper object for this dataset
aq = bq.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
#Task1: Which countries use a unit other than ppm to measure any type of pollution?
taskone = """SELECT Country 
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE unit != "ppm"
          """
#Estimating the size of query
aq.estimate_query_size(taskone)
#Query to Pandas dataframe tranformation
notppm = aq.query_to_pandas_safe(taskone)
#Solution (Task1)
notppm.Country.value_counts()

#The top 5 countries that use a unit other than ppm to measure pollution are:
#France, Spain, Germany, United States, Austria
notppm.Country.value_counts().head()

#Task2:  Which pollutants have a value of exactly 0?
tasktwo = """SELECT pollutant 
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE value = 0
          """
#Estimating the size of query
aq.estimate_query_size(tasktwo)
#Query to Pandas dataframe tranformation
value = aq.query_to_pandas_safe(tasktwo)
#Solution (Task2)
value.pollutant.value_counts()

#Top 5 pollutants having value exactly 0 are:
value.pollutant.value_counts().head()
