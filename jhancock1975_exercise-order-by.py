# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.query_to_pandas_safe("""SELECT COUNT(consecutive_number), 
  EXTRACT(HOUR FROM timestamp_of_crash)
  FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
  GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
  ORDER BY COUNT(consecutive_number) DESC
  """)
accidents.head('accident_2015')
df=accidents.query_to_pandas_safe("""
SELECT COUNT(1), first_harmful_event_name 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY first_harmful_event_name ORDER BY first_harmful_event_name
""")