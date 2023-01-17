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
from google.cloud import bigquery
client = bigquery.Client()
query = client.query(""" select count(consecutive_number),
                            EXTRACT(hour from timestamp_of_crash)
                            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                            group by EXTRACT(hour from timestamp_of_crash)
                            order by count(consecutive_number) desc
                        """)
print(query.result().to_dataframe())


query2 = client.query("""select registration_state_name, count(registration_state_name)
                        from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                        where hit_and_run LIKE '%Y%'
                        group by (registration_state_name)
                        order by count(registration_state_name) desc
                        
""")

print(query2.result().to_dataframe())