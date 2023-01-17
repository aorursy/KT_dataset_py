# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which happen by the hour of the day
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query)
accidents_by_hour
# query to find out the number of accidents which happen by the hour of the day
query = """SELECT state_number, 
                  hit_and_run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        """

per_state_hit_and_runs = accidents.query_to_pandas_safe(query)
per_state_hit_and_runs
# query to find out the number of accidents which happen by the hour of the day
query = """SELECT state_number, 
                  count(hit_and_run) as count_hr
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY state_number, hit_and_run
            HAVING hit_and_run = "Yes"
            ORDER BY count_hr DESC
        """

per_state_hit_and_runs = accidents.query_to_pandas_safe(query)
# https://en.wikipedia.org/wiki/List_of_U.S._state_abbreviations has state_number to name listing
# 6 - CA, 12 - FL, 48 - TX, 36 - NY, 13 - GA
per_state_hit_and_runs