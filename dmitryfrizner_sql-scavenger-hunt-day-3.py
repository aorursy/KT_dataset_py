import bq_helper
# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(
    active_project="bigquery-public-data",
    dataset_name="nhtsa_traffic_fatalities"
)
query_accsby_hours2015 = """
    SELECT EXTRACT(HOUR FROM timestamp_of_crash) as Hour, COUNT(consecutive_number) AS Accidents
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
    GROUP BY hour
    ORDER BY Accidents DESC
""" 
accsby_hours2015 = accidents.query_to_pandas_safe(query_accsby_hours2015)
accsby_hours2015
query_accsby_states2015 = """
    SELECT registration_state_name AS State, COUNT(`hit_and_run`) AS HRAccidents
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
    WHERE `hit_and_run`='Yes'
    GROUP BY STATE
    ORDER BY HRAccidents DESC
""" 

accsby_states2015 = accidents.query_to_pandas_safe(query_accsby_states2015)
accsby_states2015