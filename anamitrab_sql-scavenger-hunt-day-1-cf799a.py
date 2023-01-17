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
non_ppm_units = """ SELECT DISTINCT COUNTRY 
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE UNIT != 'ppm' """
countries_non_ppm = open_aq.query_to_pandas_safe(non_ppm_units)
print("No. of countries where pollutant units are not ppm : " + str(countries_non_ppm.size))
print(countries_non_ppm)
# Which pollutants have a value of exactly 0?
pollutants_is_zero = """ SELECT DISTINCT pollutant 
                         FROM `bigquery-public-data.openaq.global_air_quality`
                         WHERE value = 0
                        """
zero_pollutants = open_aq.query_to_pandas_safe(pollutants_is_zero)
print("No. of pollutants with a value of 0 : " + str(zero_pollutants.size))
print(zero_pollutants)