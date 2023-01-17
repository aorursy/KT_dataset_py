# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
#accidents.list_tables()
accidents.head('accident_2015')
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
# Your code goes here :)
query1 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash),
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash), EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query1)
accidents_by_hour.head()
plt.plot(accidents_by_hour.f1_+accidents_by_hour.f2_/24., accidents_by_hour.f0_, 'o')
plt.xticks((1,2,3,4,5,6,7,8), ('Su', 'M', 'Tu', 'W', 'Th', 'F', 'Sa'))
plt.xlabel('Day of week, hour of day')
plt.ylabel('number of accidents')
print(accidents_by_hour[0:10])
schema = accidents.table_schema('vehicle_2016')
#help(schema[0])
for col in schema:
    print(col.name, ':', col.field_type, ',', col.description, '\n')
#print(schema)
accidents.head('vehicle_2016')
query2 = """SELECT registration_state_name, COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """
hitrun_by_state = accidents.query_to_pandas_safe(query2)
hitrun_by_state.head()
