# import package with helper functions
import bq_helper
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
# print all the tables in this dataset
open_aq.list_tables()
# Which countries use a unit other than ppm to measure any type of pollution
# ordered by country to check the distinct
print("The countries that use a unit other than ppm to measure any type of pollution are:")
query_ppm = """ SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """
unit_not_ppm = open_aq.query_to_pandas_safe(query_ppm)
for country in unit_not_ppm.country.unique(): print (country)


# Which pollutants have a value of exactly zero
# ordered by country to check the distinct
print("The pollutants that have a value of zero are:")
query_value = """ SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
        """
value_zero = open_aq.query_to_pandas_safe(query_value)
for pollutant in value_zero.pollutant.unique():print(pollutant)