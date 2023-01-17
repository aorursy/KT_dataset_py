# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

#accidents.head("damage_2016")

query = """ SELECT COUNT(consecutive_number) AS Crashes, EXTRACT (HOUR FROM timestamp_of_crash) AS Time_24_hr
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
        GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)"""

prangs = accidents.query_to_pandas_safe(query)

prangs

