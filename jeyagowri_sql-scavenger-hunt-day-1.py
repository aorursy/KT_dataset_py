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
# Day1 (SQL hunt) question Which countries use a unit 
#other than ppm to measure any type of pollution?
#(Hint: to get rows where the value isn't something, use "!=")


Qry_countryunit_notppm = """
            SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
Pd_unitnoppm = open_aq.query_to_pandas_safe(Qry_countryunit_notppm)

#print total number of countries use unit other than ppm
print("Total countries unit other than ppm is - ", str(Pd_unitnoppm.size))

#list first 5 countries uses unit other than ppm
print(" First 5 Countries  unit other than ppm - \n", str(Pd_unitnoppm.country.head()))
#open_aq.estimate_query_size(countryunit_notppm_query)

# Day1 (SQL hunt) question :Which pollutants have a value of exactly 0?
Qry_poll_val0 ="""
            SELECT distinct pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0
        """
#load into panda safe mode
pd_poll_val0 = open_aq.query_to_pandas_safe(Qry_poll_val0)

#print number of pollutants value is zero
print("Total pollutants  value exactly zero is  - ", str(pd_poll_val0.size))


print("First 5 Pollutants values zero is - \n", pd_poll_val0)

