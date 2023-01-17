import numpy as np
import pandas as pd
import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
country_query = """
    SELECT DISTINCT country
    FROM `bigquery-public-data.openaq.global_air_quality`
"""

countries = open_aq.query_to_pandas_safe(country_query)
print('Number of countries:', countries.shape[0])
countries.country.head()
def select_city(country):
    return """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = '{0}'
        """.format(country)
us_cities = open_aq.query_to_pandas_safe(select_city('US'))
gb_cities = open_aq.query_to_pandas_safe(select_city('GB'))
print(us_cities.city.value_counts().head())
print(gb_cities.city.value_counts().head())
non_ppm_countries_query = """
    SELECT country
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit != 'ppm'
    GROUP BY country
"""

non_ppm_countries = open_aq.query_to_pandas_safe(non_ppm_countries_query)
non_ppm_countries.country.head()
zero_pollutants_query = """
    SELECT pollutant, city, country 
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE value = 0
    GROUP BY city, country, pollutant
"""

zero_pollutants = open_aq.query_to_pandas_safe(zero_pollutants_query)
zero_pollutants
multiple_unit_countries_query = """
    SELECT country, COUNT(DISTINCT(unit)) AS num_units
    FROM `bigquery-public-data.openaq.global_air_quality`
    GROUP BY country
    HAVING COUNT(DISTINCT(unit)) > 1
"""

multiple_unit_countries = open_aq.query_to_pandas_safe(multiple_unit_countries_query)
print(multiple_unit_countries.shape[0])
multiple_unit_countries
multiple_unit_countries_query = """
    SELECT country, unit
    FROM (
        SELECT DISTINCT country, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        ) as country_unit
    GROUP BY country, unit
    HAVING COUNT(*) > 1
"""

multiple_unit_countries = open_aq.query_to_pandas_safe(multiple_unit_countries_query)
print(multiple_unit_countries.shape[0])
multiple_unit_countries[multiple_unit_countries.country == 'FR']
result = open_aq.query_to_pandas_safe("SELECT DISTINCT country, unit FROM `bigquery-public-data.openaq.global_air_quality`")
result[result.country == 'CL']
