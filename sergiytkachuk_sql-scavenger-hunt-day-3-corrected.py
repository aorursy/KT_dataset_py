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
import bq_helper
import matplotlib.pyplot as plt

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
by_hour_sorted_query = """SELECT EXTRACT(HOUR from timestamp_of_crash), 
                                  count(*)
                            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                            where hour_of_crash <> 99
                            GROUP BY EXTRACT(HOUR from timestamp_of_crash)
                            ORDER BY count(*) DESC
                        """
by_hour_query_2015 = """SELECT hour_of_crash, 
                                  count(*)
                            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                            where hour_of_crash <> 99
                            GROUP BY hour_of_crash
                        """
by_hour_query_2016 = """SELECT hour_of_crash, 
                                  count(*)
                            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                            where hour_of_crash <> 99
                            GROUP BY hour_of_crash
                        """
queries = {2015: by_hour_query_2015, 2016: by_hour_query_2016}
keys = [k for k in queries.keys()]

accidents_by_hour_sorted = accidents.query_to_pandas_safe(by_hour_sorted_query)
print(accidents_by_hour_sorted)

for i in keys:
    plt.plot(accidents.query_to_pandas_safe(queries[i]).f0_)
    plt.title("Number of Accidents Hourly in 2015 & 2016\n (Most to least frequent)")
    plt.legend(['2015', '2016'])
vehicle_2015_query = """SELECT registration_state_name, count(*)
                            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                            where hit_and_run = 'Yes'
                            group by registration_state_name
                            order by count(*) desc
                        """

vehicle_2016_query = """SELECT registration_state_name, count(*)
                            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                            where hit_and_run = 'Yes'
                            group by registration_state_name
                            order by count(*) desc
                        """

vehicles_2015 = accidents.query_to_pandas_safe(vehicle_2015_query)
vehicles_2016 = accidents.query_to_pandas_safe(vehicle_2016_query)

print(vehicles_2015)
print(vehicles_2016)