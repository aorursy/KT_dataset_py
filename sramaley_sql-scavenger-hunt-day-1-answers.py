# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
non_ppm_countries = open_aq.query_to_pandas_safe(query)

non_ppm_array = non_ppm_countries.country.unique()
print(non_ppm_array, non_ppm_array.size)
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """
no_pollutants = open_aq.query_to_pandas_safe(query)
no_pollutants_vc = no_pollutants.pollutant.value_counts()
print(no_pollutants_vc)
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'ppm'
        """
ppm_countries = open_aq.query_to_pandas_safe(query)
ppm_array = ppm_countries.country.unique()
print(ppm_array, ppm_array.size)
import numpy as np
never_ppm_array = np.setdiff1d(non_ppm_array, ppm_array)
print(never_ppm_array, never_ppm_array.size)
query = """SELECT unit
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
units = open_aq.query_to_pandas_safe(query)
units.unit.value_counts()
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
pollutants = open_aq.query_to_pandas_safe(query)
pollutants_vc = pollutants.pollutant.value_counts()
print(pollutants_vc)
no_pollutants_vc.div(pollutants_vc)