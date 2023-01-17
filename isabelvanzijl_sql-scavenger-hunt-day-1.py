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

# query1: Which countries use a unit other than ppm to 
# measure any type of pollution?

query1 = """select distinct country
            from `bigquery-public-data.openaq.global_air_quality`
            where unit != 'ppm'
        """

non_ppm_countries = open_aq.query_to_pandas_safe(query1)

country_list = [row[0] for i, row in non_ppm_countries.iterrows()]
    
print(sorted(country_list))
print('Number of countries:', len(country_list))
# query2: Which pollutants have a value of exactly 0?

query2 = """select distinct pollutant
            from `bigquery-public-data.openaq.global_air_quality`
            where value = 0
        """

pollutants_value_zero = open_aq.query_to_pandas_safe(query2)

pollutants_list = [row[0] for i, row in pollutants_value_zero.iterrows()]
#for i, row in pollutants_value_zero.iterrows():
#    pollutants_list.append(row[0])

print(sorted(pollutants_list))
print('Number of pollutants with zero value:', len(pollutants_list))