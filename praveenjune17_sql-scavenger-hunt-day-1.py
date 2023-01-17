

# Any results you write to the current directory are saved as output.

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
us_cities['city'].value_counts().head()
query_1 = '''SELECT country
           FROM `bigquery-public-data.openaq.global_air_quality`   
           WHERE pollutant != 'ppm'
    '''
countries_not_ppm = open_aq.query_to_pandas_safe(query_1)
countries_not_ppm['country'].value_counts().head()
query_2 = '''SELECT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`   
           WHERE value = 0.00
    '''


zero_pollutant = open_aq.query_to_pandas_safe(query_2)
zero_pollutant['pollutant'].value_counts().head()




