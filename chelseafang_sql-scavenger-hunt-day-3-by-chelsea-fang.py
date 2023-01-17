import bq_helper as bq
accidents=bq.BigQueryHelper(active_project="bigquery-public-data",
                           dataset_name="nhtsa_traffic_fatalities")
accidents.head('accident_2015')
query1="""
          SELECT COUNT(consecutive_number),
                 EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
          GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
          ORDER BY COUNT(consecutive_number) DESC
"""
result1=accidents.query_to_pandas_safe(query1)
print(result1)
import matplotlib.pyplot as plt
plt.plot(result1.f1_)
plt.title("No. of Accidents by Rank of Day \n (Most to least dangerous)")
query2="""
       SELECT COUNT(consecutive_number),
              EXTRACT(HOUR FROM timestamp_of_crash)
       FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
       GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
       ORDER BY COUNT(consecutive_number) DESC
"""
result2=accidents.query_to_pandas_safe(query2)
print(result2)
plt.plot(result2.f1_)

accidents.head("vehicle_2015")
query3="""
            SELECT  registration_state_name, COUNT(hit_and_run) 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run='Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
"""
result3=accidents.query_to_pandas_safe(query3)
print(result3)
