# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# define first query
queryOne = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# use the very useful query_to_pandas_safe to avoid scanning too much data
ppmNot = open_aq.query_to_pandas_safe(queryOne)

# display returned data
ppmNot
queryTwo = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
noPollute = open_aq.query_to_pandas_safe(queryTwo)
        
noPollute
queryThree = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
        
noPolluteCounts = open_aq.query_to_pandas_safe(queryThree)
        
# count the number of times each pollutant hit that magic zero figure
noPolluteCounts.pollutant.value_counts()