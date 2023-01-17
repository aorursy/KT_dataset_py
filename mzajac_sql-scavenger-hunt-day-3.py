# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query1 = """SELECT  EXTRACT(HOUR FROM timestamp_of_crash) as HourOfAccident,
                    COUNT(consecutive_number) as NbAccidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY HourOfAccident
            ORDER BY NbAccidents DESC
        """

accidents_per_hour = accidents.query_to_pandas_safe(query1)

print('How many accidents occurred in each hour of the day in 2016?')
print (accidents_per_hour)


query2 = """SELECT  registration_state_Name as RegistrationStateName,
                    COUNTIF(hit_and_run='Yes') as NbOfHitAndRun,
                    (SELECT SUM(number_of_motor_vehicles_in_transport_mvit)
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` a
                        WHERE hit_and_run='Yes' AND a.registration_state_Name=b.registration_state_Name
                        GROUP BY registration_state_Name) as NbVehiclesInvolved
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` b
            GROUP BY RegistrationStateName
            HAVING NbOfHitAndRun >0
            ORDER BY NbOfHitAndRun DESC
        """

hit_and_run_per_state = accidents.query_to_pandas_safe(query2)

print('Which state has the most hit and runs?')
print (hit_and_run_per_state)