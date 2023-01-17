# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# Your Code Here
query="""SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number)
           """
accidents.query_to_pandas_safe(query)
accidents.head("vehicle_2016")

# Your Code Here
accidents.head("vehicle_2016")
querry="""SELECT COUNT("consecutive_number"),registration_state_name 
          FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
          GROUP BY registration_state_name
          HAVING hit_and_run="YES"
        """
accidents.query_to_pandas_safe(query)