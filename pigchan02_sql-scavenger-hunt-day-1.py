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
### Question 1

# I am defining the specific query 

MyQuery1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# just to be safe :P
countries_notppm = open_aq.query_to_pandas_safe(MyQuery1)

# now we have the answer with this array 
country_names_q1 = countries_notppm.country.unique() 
country_names_q1


# Q1: Which countries use a unit other than ppm to measure any type of pollution?
# 'AD', 'AE', 'AR', 'AT', 'AU', 'BA', 'BD', 'BE', 'BH', 'BR', 'CA', 'CH',
# 'CL', 'CN', 'CO', 'CZ', 'DE', 'DK', 'ES', 'ET', 'FI', 'FR',
# 'GB', 'GH', 'GI', 'HK', 'HR', 'HU', 'ID', 'IE', 'IL', 'IN', 'IT', 
#'KW', 'LK', 'LT', 'LU', 'LV', 'MK', 'MN', 'MT', 'MX', 'NG', 'NL',
# 'NO', 'NP', 'PE', 'PH', 'PL', 'PT', 'RS', 'RU', 'SE', 'SG', 'SI', 'SK',
#'TH', 'TR', 'TW', 'UG', 'US', 'VN', 'XK', 'ZA'



### Question 2

# again, I am defining the specific query 

MyQuery2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# just in case, i am chicken :S 
countries_zerop = open_aq.query_to_pandas_safe(MyQuery2)

# now we got the answer with this array 
country_names_q2 = countries_zerop.pollutant.unique() 
country_names_q2
#Q2: Which pollutants have a value of exactly 0? 'bc', 'co', 'o3', 'no2', 'so2', 'pm10', 'pm25'