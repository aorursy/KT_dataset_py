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
non_ppm_query = """SELECT country, unit
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE unit != 'ppm'
                """
non_ppm_measurements = open_aq.query_to_pandas_safe(non_ppm_query)
non_ppm_measurements.head()
# Let's see which countries they are
non_ppm_countries = non_ppm_measurements.country.unique()
non_ppm_countries
# And how about the other units used?
non_ppm_measurements.unit.unique()
zero_pollutant_query = """SELECT pollutant
                            FROM `bigquery-public-data.openaq.global_air_quality`
                            WHERE value = 0
                        """
zero_pollutant_measurements = open_aq.query_to_pandas_safe(zero_pollutant_query)
zero_pollutant_measurements.head()
zero_pollutant_measurements.pollutant.unique()
# Let's see which pollutants are most usually measured at zero?
zero_pollutant_measurements.groupby("pollutant").size().sort_values(ascending=False)