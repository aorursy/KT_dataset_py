# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")





#query to identify the hours in which most accidents occur
#Using the accidents_2016 table (can also use accidents_2015 table)
query = """ SELECT COUNT(consecutive_number) AS accident_count, 
                   EXTRACT(HOUR FROM timestamp_of_crash) AS hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY hour
            ORDER BY accident_count DESC"""

#query will execute only if its less than 1GB
accidents_frq_by_hour = accidents.query_to_pandas_safe(query)

#display the results
accidents_frq_by_hour

#Bar Plot - to see the data visually
import matplotlib.pyplot as plt
import seaborn as sns

plot = sns.barplot(x="hour", y="accident_count", data=accidents_frq_by_hour, color="salmon")
#query to see which state has most number of "hit and runs"
#Using the vehicle_2016 table (can also use vehicle_2015 table)
query1 = """SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
            """

#query will execute only if its less than 1GB
hit_run_by_state = accidents.query_to_pandas_safe(query1)

#display the results
hit_run_by_state
#Bar Plot - to see the data visually
import matplotlib.pyplot as plt
import seaborn as sns

f, axis = plt.subplots(figsize=(7, 14))  #Adjusting the figure size
plot = sns.barplot(x="f0_", y="registration_state_name", data=hit_run_by_state, palette="Reds_r") 

