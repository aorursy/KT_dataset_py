# borrowed from Rachael Tatman 
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query = """ SELECT EXTRACT(HOUR FROM timestamp_of_crash) as hour,
                   count(consecutive_number) as cnt
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP by 1
            order by 2 desc
        """ 

acc = accidents.query_to_pandas_safe(query)

acc
query = """ SELECT registration_state_name, count(distinct consecutive_number) as cnt_crashes,
                   count(consecutive_number) as cnt_crash_reports,
                   sum(number_of_motor_vehicles_in_transport_mvit) as veh_involved
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY 1
            order by 4 desc
        """ 

top_har_state = accidents.query_to_pandas_safe(query)

top_har_state
query = """SELECT registration_state_name, cnt_crashes,
           rank() over (order by cnt_crashes desc) as wreck_ranking
           FROM
               (SELECT registration_state_name,
                       count(distinct consecutive_number) as cnt_crashes
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                WHERE hit_and_run = 'Yes'
                GROUP BY 1
               )
                
        """ 

top_har_state = accidents.query_to_pandas_safe(query)

top_har_state
