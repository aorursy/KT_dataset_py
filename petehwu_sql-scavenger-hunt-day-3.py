# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents.estimate_query_size(query)


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# query to find out the number of accidents which 
# happen on each day of the week
#query = """SELECT COUNT(*) AS num_accidents, 
#                  EXTRACT(HOUR FROM timestamp_of_crash) as hour_of_day,
#                  EXTRACT(DAYOFWEEK from timestamp_of_crash) as day_of_week
#            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
#            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash), EXTRACT(DAYOFWEEK from timestamp_of_crash) 
#            ORDER BY COUNT(*) DESC
#        """
query = """SELECT COUNT(*) AS num_accidents, 
                EXTRACT(HOUR FROM timestamp_of_crash) crash_hour
                   
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY crash_hour
            ORDER BY  COUNT(*) DESC
        """
accidents.estimate_query_size(query)

accidents_by_hour = accidents.query_to_pandas_safe(query)
accidents_by_hour.head(24)
from matplotlib import pyplot as plt
plt.bar(x = accidents_by_hour['crash_hour'], height = accidents_by_hour['num_accidents'])
query = """
    with tb1 as (select distinct state_number, state_name 
                    from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`)
    select count(*) as hit_run, a.state_number, tb1.state_name
           from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` a,
           tb1
       where 
       a.hit_and_run = 'Yes'
       and a.state_number = tb1.state_number
       group by a.state_number, tb1.state_name
       order by hit_run desc

   """
accidents.estimate_query_size(query)
hit_and_run = accidents.query_to_pandas_safe(query)
hit_and_run.head(50)
plt.barh(y=hit_and_run['state_name'], width = hit_and_run['hit_run'])
query = """
    select count(*) as hit_run, a.registration_state_name
           from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` a
       where 
       a.hit_and_run = 'Yes'
       
       group by a.registration_state_name
       order by hit_run desc

   """
accidents.estimate_query_size(query)
reg_hit_and_run = accidents.query_to_pandas_safe(query)
reg_hit_and_run
plt.barh(y = reg_hit_and_run['registration_state_name'], width = reg_hit_and_run['hit_run'])
