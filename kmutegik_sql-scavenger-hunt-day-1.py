#Import bq helper library
import bq_helper
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
#Let's see the tables in the dataset
open_aq.list_tables() #turns out there is one table
#Lets see how a few rows in thetable looks like
open_aq.head("global_air_quality")
country_query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
no_ppm_country = open_aq.query_to_pandas_safe(country_query)#data frame of the countries
no_ppm_country_array = no_ppm_country.country.unique()#to get unique country codes without repetitions.
print(no_ppm_country_array, no_ppm_country_array.size)
pollutants_query=""" SELECT pollutant
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value=0
            """
pollutants = open_aq.query_to_pandas_safe(pollutants_query)
pollutant_array=pollutants.pollutant.unique()#unique pollutants
print(pollutant_array,pollutant_array.size)
pollutants_query=""" SELECT DISTINCT pollutant
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value=0
            """
pollutants = open_aq.query_to_pandas_safe(pollutants_query)
pollutants