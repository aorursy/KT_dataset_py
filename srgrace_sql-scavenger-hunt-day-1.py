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

# Which countries use a unit other than ppm to measure any type of pollution?
query = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# run query
not_ppm_countries = open_aq.query_to_pandas_safe(query)
not_ppm_countries.head()

# remove duplicates

not_ppm_countries.drop_duplicates(subset='country', inplace=True)
not_ppm_countries.head()


# print countries that use a unit other than ppm to measure any type of pollution

not_ppm_countries_list = not_ppm_countries.country.tolist()
print('There are {} countries that use a unit other than ppm to measure any type of pollution:'
      .format(len(not_ppm_countries_list)))

print(not_ppm_countries_list)

query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
# run query

zero_pollutants = open_aq.query_to_pandas_safe(query)
zero_pollutants.head()
# remove duplicates

zero_pollutants.drop_duplicates(subset='pollutant', inplace=True)
zero_pollutants.head()

# print pollutants which have a value of exactly 0

zero_pollutants_list = zero_pollutants.pollutant.tolist()
print('There are {} pollutants which have a value of exactly equals to 0:'.format(len(zero_pollutants)))

print(zero_pollutants_list)
