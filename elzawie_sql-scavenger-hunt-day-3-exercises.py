# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
print("The dataset consists of the below {} tables:".format(len(accidents.list_tables())))
accidents.list_tables()
print("The table consists of the below {} attributes:".format(len(accidents.table_schema("accident_2016"))))
accidents.table_schema("accident_2016")
# query to find out the number of accidents which 
# happen on each day of the week
query_1 = """SELECT COUNT(consecutive_number) AS Counts, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS Hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY Hour
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if it would use too much of our quota,
# with the limit set to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query_1)
accidents_by_hour.head(10)
query_2 = """ SELECT registration_state_name AS RegistrationState, COUNT(consecutive_number) AS Count 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            GROUP BY RegistrationState
            ORDER BY COUNT(consecutive_number) DESC
          """
accidents.estimate_query_size(query_2)
hitrun_counts = accidents.query_to_pandas_safe(query_2)
hitrun_counts.head(10)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,13))
plt.bar(hitrun_counts.RegistrationState[:8], hitrun_counts.Count[:8], color = 'green')
plt.show()