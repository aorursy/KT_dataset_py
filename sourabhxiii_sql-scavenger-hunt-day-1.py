# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.head("global_air_quality")
# countries use a unit other than ppm to measure the pollution level
query = """ SELECT DISTINCT country FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'
            ORDER BY 1
        """
# run the query, make sure that the returned result set is not more than 1gig (by default) in size
countries_not_using_ppm_unit = open_aq.query_to_pandas_safe(query)
# print the result
print(countries_not_using_ppm_unit)
# answer to the 1st interpretation listed above
query = """ SELECT DISTINCT(pollutant), country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY 1, 2
        """
# run the query, make sure that the returned result set is not more than 1gig (by default) in size
zero_value_pollutant = open_aq.query_to_pandas_safe(query)
# print the result
print(zero_value_pollutant)
# answer to the 2nd interpretation listed above
query = """ SELECT country, pollutant, SUM(value) AS per_country_per_pollutant_total
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY country, pollutant
            HAVING per_country_per_pollutant_total = 0
            ORDER BY 1, 2
        """
# run the query, make sure that the returned result set is not more than 1gig (by default) in size
within_a_country_zero_value_pollutant = open_aq.query_to_pandas_safe(query)
# print the result
print(within_a_country_zero_value_pollutant)
# answer to the 3rd interpretation listed above
# BUT THIS RETURNS MORE THAN 1gig OF DATA!
query = """ SELECT pollutant, SUM(value) AS pollutant_total
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY pollutant
            HAVING pollutant_total = 0
            ORDER BY 1
        """
# run the query, make sure that the returned result set is not more than 1gig (by default) in size
zero_value_pollutant = open_aq.query_to_pandas_safe(query)
# print the result
print(zero_value_pollutant)
