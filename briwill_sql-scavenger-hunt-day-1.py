# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# query to select all cities where
# unit is not equal to PPM
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'
            ORDER BY country
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
non_ppm_countries = open_aq.query_to_pandas_safe(query)

# display query results
non_ppm_countries
# query to select all pollutants that have had a reading where pollutant value
# where pollutant value was equal to zero
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
        """

zero_pol_readings = open_aq.query_to_pandas_safe(query)

# display query results
zero_pol_readings
# query to select pollutants (if any) where total value across
# all readings is equal to zero
query = """SELECT pollutant, SUM(value) as tot_value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'LT'
            GROUP BY pollutant
            HAVING tot_value = 0
        """
zero_pol_readings2 = open_aq.query_to_pandas_safe(query)

# display query results
zero_pol_readings2