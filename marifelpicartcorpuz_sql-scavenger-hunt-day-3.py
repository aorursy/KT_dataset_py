# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# print the first five rows of the "full" table
accidents.head("accident_2016")
#how many accidents occurred in each hour of the day in 2015,
#sorted by the the number of accidents which occurred each day

query1 = """SELECT  
                    EXTRACT(DATE FROM T1.timestamp_of_crash) AS ACCIDENT_DATE, 
                    EXTRACT(HOUR FROM T1.timestamp_of_crash) AS ACCIDENT_HOUR, 
                    COUNT(T1.consecutive_number) AS CT_ACCDTS_BYHOUR_EACHDAY,
                    SUM(T2.CT_ACCDTS_EACHDAY) AS CT_ACCDTS_EACHDAY
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` T1
            
            LEFT JOIN (SELECT EXTRACT(DATE FROM timestamp_of_crash) AS ACCIDENT_DATE, 
                              COUNT(consecutive_number) AS CT_ACCDTS_EACHDAY 
                       FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                       GROUP BY 1) T2
                       
            ON EXTRACT(DATE FROM T1.timestamp_of_crash) = T2.ACCIDENT_DATE
            GROUP BY 1,2
            ORDER BY 4
        """
# Estimate query size
accidents.estimate_query_size(query1)
# cancel the query if it exceeds 1 GB
HOUR_accidents = accidents.query_to_pandas_safe(query1)
print("number of accidents by hour and day")
print(HOUR_accidents)
# print the first five rows of the "full" table
accidents.head("vehicle_2015")
#Which state has the most hit and runs?
#Return a table with the number of vehicles registered in each state 
#that were involved in hit-and-run accidents, sorted by the number of hit-and-run accidents.
#Use the vehicle_2015 or vehicle_2016 table for this, 
#especially the registration_state_name and hit_and_run columns.


query2 = """SELECT  registration_state_name, sum(vehicle_number) as tot_vehicles
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` T1
            where hit_and_run = 'Yes'
            group by 1
            order by 2
        """

# Estimate query size
accidents.estimate_query_size(query2)
# cancel the query if it exceeds 1 GB
Hit_n_Run = accidents.query_to_pandas_safe(query2)
print("number of hit and run by state")
print(Hit_n_Run)