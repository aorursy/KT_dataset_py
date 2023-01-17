# import package with helper functions 
import bq_helper
import matplotlib.pyplot as plt

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.table_schema("accident_2016")
accidents.head("accident_2016", num_rows=10)
query = """SELECT COUNT(consecutive_number),
                EXTRACT(HOUR from timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR from timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accident_count = accidents.query_to_pandas_safe(query)
# sort by hour of day instead, it makes more sense for plotting
query = """SELECT COUNT(consecutive_number),
                EXTRACT(HOUR from timestamp_of_crash) AS time
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY time
            ORDER BY time
        """
accident_count = accidents.query_to_pandas_safe(query)
print(accident_count)
plt.plot(accident_count.time, accident_count.f0_)
plt.show()
accidents.head("vehicle_2016", num_rows=10, selected_columns="hit_and_run")
query = """SELECT COUNT(hit_and_run) AS hit_run,
                registration_state_name AS registration_state                
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
hit_run_count = accidents.query_to_pandas_safe(query)
print(hit_run_count)