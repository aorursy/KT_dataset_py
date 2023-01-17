# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.head("accident_2015")
# Your Code Here
query = """select count(consecutive_number),
        extract (hour from timestamp_of_crash)
        from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        group by extract(hour from timestamp_of_crash)
        order by count(consecutive_number) desc"""

output = accidents.query_to_pandas_safe(query)
import matplotlib.pyplot as plt
plt.plot(output.f0_)
plt.title("Number of accidents by rank of hour \n (Most to least dangerous)")
print(output)
accidents.head("vehicle_2015")
# Your Code Here
query = """select count(consecutive_number),
            state_number
        from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        where hit_and_run = "Yes"
        group by state_number
        order by count(state_number) desc

"""
output = accidents.query_to_pandas_safe(query)
print(output)