# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
# Your Code Here
query = """

SELECT * 
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
 

""" 

res = accidents.query_to_pandas_safe(query)
print(res.head())
# Your Code Here
query = """

SELECT EXTRACT(HOUR
  FROM timestamp_of_crash), count(timestamp_of_crash)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
group by EXTRACT(HOUR
  FROM timestamp_of_crash) 
  order by count(timestamp_of_crash) desc

""" 

res = accidents.query_to_pandas_safe(query)
print(res.head())
# Your Code Here

# Your Code Here
query = """

SELECT 
distinct hit_and_run 
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` 
 

""" 

res = accidents.query_to_pandas_safe(query)
print(res.head())

# Your Code Here

# Your Code Here
query = """

SELECT *
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` 
 limit 1

""" 

res = accidents.query_to_pandas_safe(query)
print(res.head())
# Your Code Here

# Your Code Here
query = """

SELECT 
registration_state_name, count(timestamp_of_crash)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` 
 where hit_and_run='Yes'
 group by registration_state_name
order by count(timestamp_of_crash) desc
 
""" 

res = accidents.query_to_pandas_safe(query)
print(res.head())
