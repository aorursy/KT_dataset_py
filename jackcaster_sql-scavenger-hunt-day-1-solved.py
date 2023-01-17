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
#Utility function for checking the cost of the query
def control_query_size(database, query, limit_cost=0.1):
    if database.estimate_query_size(query) <= limit_cost:
        print("OK to run query")
    else:
        raise ValueError('The query is too expensive.')
# Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

control_query_size(database=open_aq, query=query, limit_cost=0.1)

countries_no_ppm = open_aq.query_to_pandas_safe(query)
countries_no_ppm.head()
print('There are {} countries that do not use ppm as unit'.format(len(countries_no_ppm.country.unique())))

for idx, country in enumerate(countries_no_ppm.country.unique()):
    print('{idx} \t {country}'.format(idx=idx+1, country=country))
# Which pollutants have a value of exactly 0?
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """

control_query_size(database=open_aq, query=query, limit_cost=0.1)
pollutants_w_value_0 = open_aq.query_to_pandas_safe(query)
pollutants_w_value_0.head()
for idx, pollutant in enumerate(pollutants_w_value_0.pollutant.unique()):
    print('{idx} \t {pollutant}'.format(idx=idx+1, pollutant=pollutant))

print('There are {} pollutant with null value'.format(len(pollutants_w_value_0.pollutant.unique())))