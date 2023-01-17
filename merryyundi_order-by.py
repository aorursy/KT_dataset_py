# import package with helper functions 

import bq_helper



# create a helper object for this dataset

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                     dataset_name="nhtsa_traffic_fatalities")



# print the first couple rows of the "accident_2015" table

accidents.head("accident_2015")
# query to find out the number of accidents which 

# happen on each day of the week

query = """SELECT COUNT(consecutive_number) AS num_accidents, 

                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY day_of_week

            ORDER BY num_accidents DESC

        """
# the query_to_pandas_safe method will cancel the query if

# it would use too much of your quota, with the limit set 

# to 1 GB by default

accidents_by_day = accidents.query_to_pandas_safe(query)
print(accidents_by_day)
# library for plotting

import matplotlib.pyplot as plt



# make a plot to show that our data is, actually, sorted:

plt.plot(accidents_by_day.num_accidents)

plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")

plt.show()