#Part 1: Which countries use a unit other than ppm to measure any type of pollution?
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
query1 = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'
            ORDER BY country
        """
nonppm_countries = open_aq.query_to_pandas_safe(query1)
nonppm_countries
#Part 2: Which pollutants have a value of exactly 0
query2 = """SELECT DISTINCT pollutant,value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
        """
zero_pollutants = open_aq.query_to_pandas_safe(query2)
zero_pollutants
