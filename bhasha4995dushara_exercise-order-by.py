# import package with helper functions 
import matplotlib.pyplot as plt
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
#accidents.list_tables()

accidents.head("accident_2015",5)
# Your Code Here
query = '''select count(consecutive_number),extract(HOUR from timestamp_of_crash) from 
            `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
                group by extract(HOUR from timestamp_of_crash)
                order by count(consecutive_number) desc'''
a_hour = accidents.query_to_pandas_safe(query)

plt.plot(a_hour.f1_,a_hour.f0_)
plt.title("Number of Accidents by Rank of Hour")
plt.show()

accidents.table_schema("vehicle_2015")
accidents.head("vehicle_2015",5)
query = ''' select count(hit_and_run),registration_state_name from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            where hit_and_run = "Yes"
            group by registration_state_name
            order by count(hit_and_run) desc'''
a_state = accidents.query_to_pandas_safe(query)
a_state.head()
