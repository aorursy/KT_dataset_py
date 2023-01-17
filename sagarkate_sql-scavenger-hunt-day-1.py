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
#Which countries use a unit other than ppm to measure any type of pollution?
queryCountriesWithoutppm = """
SELECT DISTINCT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != "ppm"
"""

dfCountriesWithoutppm = open_aq.query_to_pandas_safe(queryCountriesWithoutppm)

print("\nCountries that use unit other than ppm:")
for index, row in dfCountriesWithoutppm.iterrows():
    print(row['country'])

#Which pollutants have a value of exactly 0
queryPollutantsWithValueZero = """
SELECT DISTINCT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""

dfPollutantsWithValueZero = open_aq.query_to_pandas_safe(queryPollutantsWithValueZero)

print("\nPollutants that have value as zero:")
for index, row in dfPollutantsWithValueZero.iterrows():
    print(row['pollutant'])