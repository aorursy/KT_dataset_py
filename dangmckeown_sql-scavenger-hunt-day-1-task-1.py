#SQL scavenger hunt day 1 task 1

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

open_aq.head("global_air_quality")

# query task: Which countries use a unit other than ppm to measure any type of pollution?

query = """SELECT country 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'pmm'
            GROUP BY country"""



no_pmm_countries = open_aq.query_to_pandas_safe(query)


no_pmm_countries.country
