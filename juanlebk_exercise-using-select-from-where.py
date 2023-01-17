# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head('global_air_quality')
# Query to select all the items from the "city" column where the "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'"""
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.get_values()
us_cities.describe()
us_cities.info()
us_cities.city.nunique()
us_cities.city.value_counts()
open_aq.head('global_air_quality')
query_sel_country_by_unit = """SELECT country
                                FROM `bigquery-public-data.openaq.global_air_quality`
                                WHERE unit != 'ppm'"""
countries_notuse_ppm_unit = open_aq.query_to_pandas_safe(query_sel_country_by_unit)
open_aq.table_schema('global_air_quality')
query_sel_country = """SELECT COUNTRY
                        FROM `bigquery-public-data.openaq.global_air_quality`"""
countries = open_aq.query_to_pandas_safe(query_sel_country)
countries.count()
countries_notuse_ppm_unit.count()
countries_notuse_ppm_unit.nunique()
countries_notuse_ppm_unit = countries_notuse_ppm_unit.country.unique()
countries_notuse_ppm_unit
open_aq.head('global_air_quality')
query_sel_pollutant_have_value_0 = """SELECT pollutant
                                        FROM `bigquery-public-data.openaq.global_air_quality`
                                        WHERE value = 0"""
pollutant_have_value_0 = open_aq.query_to_pandas(query_sel_pollutant_have_value_0).pollutant.unique()
query_sel_pollutant_have_value_dif0 = """SELECT pollutant
                                        FROM `bigquery-public-data.openaq.global_air_quality`
                                        WHERE value != 0"""
pollutant_have_value_dif0 = open_aq.query_to_pandas(query_sel_pollutant_have_value_dif0).pollutant.unique()
print(pollutant_have_value_0)
print(pollutant_have_value_dif0)
pollutant_have_value_exact_0 = open_aq.query_to_pandas(query_sel_pollutant_have_value_0)
pollutant_have_value_exact_0.pollutant.value_counts()