# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# Check the counries in the database 
country_query = """
        SELECT DISTINCT country
        FROM `bigquery-public-data.openaq.global_air_quality`
    """

countries = open_aq.query_to_pandas_safe(country_query)
print('Number of countries:', countries.shape[0])
#countries is a dataframe
countries.country
# query to select all the items from the "city" where the "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
us_cities = open_aq.query_to_pandas_safe(query)
# Five U.S. cities that have the most measurements taken 
us_cities.city.value_counts().head()
def select_cities(country):
    return """
            SELECT city 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = '{0}'
            """.format(country)
ca_cities = open_aq.query_to_pandas_safe(select_cities('CA'))
gb_cities = open_aq.query_to_pandas_safe(select_cities('GB'))

print(ca_cities.city.value_counts().head())
print(gb_cities.city.value_counts().head())
# 1. countries that do not use ppm to measure pollution 
# either use group by or select distinct 
non_ppm_countries_query = """SELECT country
                             FROM `bigquery-public-data.openaq.global_air_quality`
                             WHERE unit != 'ppm'
                             GROUP BY country
                          """
non_ppm_countries = open_aq.query_to_pandas_safe(non_ppm_countries_query)

# eqv to non_ppm_countries.count()
print('Number of non_ppm countries ',non_ppm_countries.shape[0] ) 
non_ppm_countries.country.head()
# 2. which pollutants have value of 0 

zero_pollutants_query = """
    SELECT count(pollutant) AS num_pollutant_cities, pollutant
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE value = 0
    GROUP BY pollutant
"""
zero_pollutants = open_aq.query_to_pandas_safe(zero_pollutants_query)
zero_pollutants

## equivilant answer to the above by using select distinct 

alt_zero_pollutants_query = """
    SELECT DISTINCT pollutant
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE value = 0
"""
alt_zero_pollutants = open_aq.query_to_pandas_safe(alt_zero_pollutants_query)
print('Number of pollutant with zero value: ', alt_zero_pollutants.pollutant.count())
alt_zero_pollutants

