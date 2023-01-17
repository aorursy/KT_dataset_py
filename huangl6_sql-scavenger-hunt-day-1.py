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
query = """SELECT country, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        """
not_ppm = open_aq.query_to_pandas_safe(query)
not_ppm.head()

#remove duplicate countries
not_ppm = not_ppm.drop_duplicates(subset='country')
not_ppm.head()
#Find total number of countries that use a unit other than ppm to measure pollution
hipsterCountries = not_ppm.country.tolist()
print(len(hipsterCountries))
print('There are {} countries that use something other than ppm to measure pollution'.format(len(hipsterCountries)))
print('These countries are: {}'.format(hipsterCountries))
query = """SELECT pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """
zeroPollutants = open_aq.query_to_pandas_safe(query)
zeroPollutants.head()
#remove duplicates
zeroPollutants = zeroPollutants.drop_duplicates(subset = 'pollutant')
zeroPollutants.head()
#Find pollutants that have value of 0
zeroPollute = zeroPollutants.pollutant.tolist()
print('There are {} pollutants with value 0.'.format(len(zeroPollute)))
print('They are: {}'.format(zeroPollute))