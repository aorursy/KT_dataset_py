# Import package with helper functions 
import bq_helper

# Create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# Print the first couple rows of the "accident_2015" table
accidents.head("accident_2015")

# Full schema of "accident_2015" tables
print(accidents.table_schema("accident_2015"))
# Question1: Which hours of the day do the most accidents occur during?
    # I try with "accident_2015" but it is the same with the table "accident_2016"

question1 = """SELECT COUNT(consecutive_number) as Accidents, timestamp_of_crash as Timestamp,
                      EXTRACT(DAY FROM timestamp_of_crash) as Day_of_2015,
                      EXTRACT(HOUR FROM timestamp_of_crash) as Hour_of_2015                      
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY timestamp_of_crash, EXTRACT(HOUR FROM timestamp_of_crash), 
                     EXTRACT(DAY FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

# Estimation of query question1
print(accidents.estimate_query_size(question1))

# I use max_db_scanned = 2 to limit at 2 GB
hour_accidents = accidents.query_to_pandas_safe(question1, max_gb_scanned=2)

# Print Dataframe Size
print('Dataframe Size: {} Bytes'.format(int(hour_accidents.memory_usage(index=True, deep=True).sum())))

# Print Dataframe "hour_accidents"
print(hour_accidents.head())
# Print the first couple rows of the "vehicle_2015" table
accidents.head("vehicle_2015")

# Full schema of "vehicle_2015" tables
print(accidents.table_schema("vehicle_2015"))
# Question2: Which state has the most hit and runs?
    # I try with "vehicle_2015" but it is the same with the table "vehicle_2016"

question2 = """SELECT state_number as State_Number, SUM(vehicle_number) as Vehicles_Registered, 
                      hit_and_run as Hit_and_Run, COUNT(hit_and_run) as Number_Hit_and_Run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY state_number, hit_and_run
            ORDER BY COUNT(hit_and_run) DESC
        """

# Estimation of query question1
print(accidents.estimate_query_size(question2))

# I use max_db_scanned = 2 to limit at 2 GB
hit_and_run = accidents.query_to_pandas_safe(question2, max_gb_scanned=2)

# Print Dataframe Size
print('Dataframe Size: {} Bytes'.format(int(hit_and_run.memory_usage(index=True, deep=True).sum())))

# Print Dataframe "stories"
print(hit_and_run.head())
# Question3: Another Possible Query

question3 = """SELECT registration_state_name as State_Name, SUM(vehicle_number) as Vehicles_Registered, 
                      EXTRACT(HOUR FROM timestamp_of_crash) as Hour_of_2016,
                      timestamp_of_crash as Timestamp
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = 'Yes'
                  and registration_state_name != 'Unknown'
            GROUP BY registration_state_name, timestamp_of_crash
            ORDER BY SUM(vehicle_number) DESC
        """

# Estimation of query question1
print(accidents.estimate_query_size(question3))

# I use max_db_scanned = 2 to limit at 2 GB
Hour_Frequency = accidents.query_to_pandas_safe(question3, max_gb_scanned=2)

# Print Dataframe "Hour_Frequency"
print(Hour_Frequency.head())