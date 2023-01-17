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
unit_not_ppm_query = """ SELECT DISTINCT COUNTRY 
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE UNIT != 'ppm' """

countries_not_ppm_unit = open_aq.query_to_pandas_safe(unit_not_ppm_query)
print("Number of countries where the pollutant measure is not ppm :" + str(countries_not_ppm_unit.size))
# print(countries_not_ppm_unit.size)


unit_ppm_query = """ SELECT DISTINCT COUNTRY 
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE UNIT = 'ppm' """

countries_ppm_unit = open_aq.query_to_pandas_safe(unit_ppm_query)

print("Number of countries where the pollutant measure is ppm :" + str(countries_ppm_unit.size))


distinct_country_query = """ SELECT DISTINCT COUNTRY 
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    """

distinct_countries = open_aq.query_to_pandas_safe(distinct_country_query)

print("Number of distinct countries in the dataset :" + str(distinct_countries.size))



# Which pollutants have a value of exactly 0?

pollutant_zero_query = """ SELECT distinct pollutant 
                         FROM `bigquery-public-data.openaq.global_air_quality`
                         WHERE value = 0
                        """
distinct_pollutant_zero = open_aq.query_to_pandas_safe(pollutant_zero_query)
print(distinct_pollutant_zero)


distinct_pollutant_query = """ SELECT distinct pollutant 
                         FROM `bigquery-public-data.openaq.global_air_quality`
                        """
distinct_pollutant = open_aq.query_to_pandas_safe(distinct_pollutant_query)
print(distinct_pollutant)
