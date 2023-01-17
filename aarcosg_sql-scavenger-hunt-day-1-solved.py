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
# query to select all countries that use a unit other
# than ppm to measure any type of pollution
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """
# run query
df_not_ppm_countries = open_aq.query_to_pandas_safe(query)
df_not_ppm_countries.head()
# print countries that use a unit other 
# than ppm to measure any type of pollution
not_ppm_countries = df_not_ppm_countries.country.tolist()
print('There are {} countries that use a unit other than ppm to measure any type of pollution:'
      .format(len(not_ppm_countries)))
print(not_ppm_countries)
# query to select all countries that use a unit other
# than ppm to measure any type of pollution
query = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# run query
df_not_ppm_countries = open_aq.query_to_pandas_safe(query)
df_not_ppm_countries.head()
# remove duplicates
df_not_ppm_countries.drop_duplicates(subset='country', inplace=True)
df_not_ppm_countries.head()
# print countries that use a unit other 
# than ppm to measure any type of pollution
not_ppm_countries = df_not_ppm_countries.country.tolist()
print('There are {} countries that use a unit other than ppm to measure any type of pollution:'
      .format(len(not_ppm_countries)))
print(not_ppm_countries)
# query to select all pollutants which have a value of exactly 0
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
            ORDER BY pollutant
        """
# run query
df_pollutants_zero = open_aq.query_to_pandas_safe(query)
df_pollutants_zero.head()
# print pollutants which have a value of exactly 0
pollutants_zero = df_pollutants_zero.pollutant.tolist()
print('There are {} pollutants which have a value of exactly 0:'
      .format(len(pollutants_zero)))
print(pollutants_zero)
# query to select all pollutants which have a value of exactly 0
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
# run query
df_pollutants_zero = open_aq.query_to_pandas_safe(query)
df_pollutants_zero.head()
# remove duplicates
df_pollutants_zero.drop_duplicates(subset='pollutant', inplace=True)
df_pollutants_zero.head()
# print pollutants which have a value of exactly 0
pollutants_zero = df_pollutants_zero.pollutant.tolist()
print('There are {} pollutants which have a value of exactly 0:'
      .format(len(pollutants_zero)))
print(pollutants_zero)