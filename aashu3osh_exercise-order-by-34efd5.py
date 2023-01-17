# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.head("accident_2015")

accidents.table_schema("accident_2015")

query = """SELECT EXTRACT(HOUR from timestamp_of_crash), COUNT(number_of_fatalities)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
ORDER BY COUNT(number_of_fatalities) DESC"""

fatalities = accidents.query_to_pandas_safe(query)
fatalities.head()
accidents.head("accident_2015",selected_columns="number_of_drunk_drivers",num_rows=10)
accidents.head("accident_2015",selected_columns="state_name",num_rows=10)

query2 = """SELECT state_name , COUNT(number_of_drunk_drivers)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY state_name
ORDER BY COUNT(number_of_drunk_drivers) DESC"""

Drunk = accidents.query_to_pandas_safe(query2)
Drunk.head()

# I couldn't find "hit_and_run" column, hence i solved for drunk drivers case.
# tell me if you find that column