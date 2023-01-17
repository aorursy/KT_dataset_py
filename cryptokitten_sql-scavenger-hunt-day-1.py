# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# Which countries use a unit other than ppm to measure any type of pollution?
query = """SELECT distinct(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
non_ppm_countries = open_aq.query_to_pandas_safe(query)

# View the countries in which pollutants are measured with a unit other than ppm
non_ppm_countries


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# Which pollutants have a value of exactly 0? Here, we assume that we are interested
# in all pollutants for which there is at least one record where value = 0.0
query = """SELECT distinct(pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

zero_level_pollutants = open_aq.query_to_pandas_safe(query)
zero_level_pollutants
