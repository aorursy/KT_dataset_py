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
# Your code goes here :)

# list tables
accidents.list_tables()

# list column names
accidents.table_schema('accident_2015')
# query for accident_2015 or accident_2016 
year_list = ['2015', '2016']
accidents_by_hour = {}
for year in year_list:
    query = """
            SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_{}`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
            """.format(year)

    accidents_by_hour.setdefault(year, accidents.query_to_pandas_safe(query))
# plot on accident_2015
accidents_by_hour['2015'].set_index('f1_')['f0_'].plot.bar()
# plot on accident_2016
accidents_by_hour['2016'].set_index('f1_')['f0_'].plot.bar()
accidents.head('vehicle_2015')
# list column names
accidents.table_schema('vehicle_2015')
# query for vehicle_2015 or vehicle_2016 
year_list = ['2015', '2016']
state_of_hit_and_run = {}
for year in year_list:
    query = """
            SELECT COUNTIF(hit_and_run = "Yes"), registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_{}`
            GROUP BY registration_state_name
            ORDER BY COUNTIF(hit_and_run = "Yes") DESC
            """.format(year)

    state_of_hit_and_run.setdefault(year, accidents.query_to_pandas_safe(query))
# show vehicle_2015
state_of_hit_and_run['2015'].head()
# show vehicle_2016
state_of_hit_and_run['2016'].head()