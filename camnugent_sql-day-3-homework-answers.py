import bq_helper

accident_dat = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accident_dat.list_tables()
test_data = accident_dat.head('accident_2015')
test_data.columns
q1= """
    SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash)
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
    GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
    ORDER BY COUNT(consecutive_number)
    """
accident_dat.estimate_query_size(q1)
accidents_by_hour = accident_dat.query_to_pandas_safe(q1)
accidents_by_hour
accidents_by_hour.columns = ["num_accidents","hour_of_day"]
accidents_by_hour = accidents_by_hour.sort_values('hour_of_day', axis=0)
import matplotlib.pyplot as plt
plt.bar(accidents_by_hour['hour_of_day'], accidents_by_hour['num_accidents'])
v_hickle =  accident_dat.head('vehicle_2015')
v_hickle['hit_and_run']
list(v_hickle.columns)
q2= """ SELECT registration_state_name, hit_and_run, COUNT(hit_and_run) 
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        GROUP BY registration_state_name, hit_and_run
        ORDER BY COUNT(hit_and_run) DESC
    """
accident_dat.estimate_query_size(q2)
hit_and_run_state = accident_dat.query_to_pandas_safe(q2)
hit_and_run_state
hits_state = hit_and_run_state[hit_and_run_state['hit_and_run']!= "No"]
hits_state