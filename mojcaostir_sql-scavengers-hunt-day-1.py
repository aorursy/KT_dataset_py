#import package with helper function
import bq_helper

#create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                 dataset_name='openaq')

#print all the tables in this dataset
open_aq.list_tables()


#print first couple of rows
open_aq.head('global_air_quality')
#query to select all the items from the "city" column where the "country" column is is"us"
query = """SELECT city
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE country = 'US'
        """

#query to pandas will only execute if it is smaller than one gigabyte
us_cities = open_aq.query_to_pandas_safe(query)

# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
         """
no_ppm_country = open_aq.query_to_pandas_safe(query1)
no_ppm_country.country.value_counts()
query2 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
         """
no_ppm_country = open_aq.query_to_pandas_safe(query2)
no_ppm_country
query3 = """SELECT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
        """
val = open_aq.query_to_pandas_safe(query3)
val.pollutant.value_counts()
query4 = """SELECT DISTINCT pollutant, city
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
           ORDER BY city
        """
val_city = open_aq.query_to_pandas_safe(query4)
val_city