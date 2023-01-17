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
# The DISTINCT keyword tells the query to ignore duplicates of the column in parentheses,
# which is 'country' in query1 and 'pollutant' in query2.

# The ORDER BY clause tells the query to sort the rows according the specified column.
# For strings, the sorting is done lexicographically (fancy version of alphabetically).

# The DISTINCT keyword is necessary for query1 because if 1 location in a 
# given country doesn't use ppm, then it is likely that other places in that country 
#do to. Without the DISTINCT keyword, the query could return a country's name 
# many times when all you care about is if it is returned at least once.
query1 = """SELECT DISTINCT(country)
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm'
           ORDER BY country
       """
results_1 = open_aq.query_to_pandas_safe(query1)
print('Number of Countries Using A Unit Other Than ppm: %s' % results_1.shape[0])
print('First 10 Countries Ordered Alphabetically')
print(results_1['country'].head(10))

# The DISTINCT keyword is necessary for query2 because a single pollutant 
# can be measured to have a value of exactly 0 at multiple locations. If you 
# don't have it, the query returns over 100 rows. If you include it,
# the query returns just the name of the pollutants with a value of 0.0 
query2 = """SELECT DISTINCT(pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
            ORDER BY pollutant
        """
results_2 = open_aq.query_to_pandas_safe(query2)
print('\nPollutants with a Value of 0.0 Ordered Alphabetically')
print(results_2['pollutant'])

