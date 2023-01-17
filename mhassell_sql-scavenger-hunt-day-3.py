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


print(accidents_by_day)
# import package with helper functions 
import bq_helper
import matplotlib.pyplot as plt

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query1 = """SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash) as hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
            GROUP BY hour
            ORDER BY hour ASC
        """

accidents_by_hour = accidents.query_to_pandas_safe(query1)

plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Hour")
# Let's validate this with the hour of crash column
query2 = """SELECT COUNT(consecutive_number), hour_of_crash as hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY hour ASC
        """

        # ORDER BY COUNT(consecutive_number) DESC  <- this seems to break it

accidents_by_hour_validate = accidents.query_to_pandas_safe(query2)

plt.plot(accidents_by_hour_validate.f0_)
plt.title("Number of Accidents by Hour (validate)")