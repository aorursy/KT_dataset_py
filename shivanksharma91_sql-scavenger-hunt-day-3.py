# Importing package helper
import bq_helper

#Creating helper object
traffic_fatalities=bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                           dataset_name='nhtsa_traffic_fatalities')
#Query to find out Number of accidents occured in each hour of day in 2015
query1='''SELECT count(consecutive_number) as Nbr_of_Accidents,
                EXTRACT(Hour from timestamp_of_crash) As Hour
                
         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
         GROUP BY 2
         order by Nbr_of_Accidents DESC
      '''
accidents_by_hour=traffic_fatalities.query_to_pandas_safe(query1)
# library for plotting
import matplotlib.pyplot as plt
plt.plot(accidents_by_hour.Nbr_of_Accidents,accidents_by_hour.Hour)
plt.title("Number of Accidents by rank Hour \n (Most to least dangerous)")
print(accidents_by_hour)

query2="""SELECT Count(registration_state) As Nbr_of_Hit_And_Run,
           registration_state_name AS State
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           WHERE hit_and_run="Yes"
           GROUP BY 2
           Order BY Nbr_of_Hit_And_Run DESC
       """

hit_run=traffic_fatalities.query_to_pandas_safe(query2)

plt.plot(hit_run.Nbr_of_Hit_And_Run)
plt.title("Number of Nbr of Hit And Run by rank State \n (Most to least dangerous)")
print(hit_run)
