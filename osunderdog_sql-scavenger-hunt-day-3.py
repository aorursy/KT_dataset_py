# import package with helper functions 

import bq_helper



import matplotlib.pyplot as plt





plt.rcParams['figure.figsize'] = [10,5]



# create a helper object for this dataset

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="nhtsa_traffic_fatalities")
query = """SELECT COUNT(consecutive_number) as cnt, 

                  EXTRACT(HOUR FROM timestamp_of_crash) as day_hour

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY day_hour

            ORDER BY cnt DESC

        """

accidents_by_hour = accidents.query_to_pandas_safe(query)

accidents_by_hour





# make a plot to show that our data is, actually, sorted:

plt.bar(accidents_by_hour.day_hour, accidents_by_hour.cnt)

plt.title("Number of Accidents by Rank of Day-Hour \n (Most to least dangerous)")



query = """select state_name, cnt

           from (

            SELECT v.state_number, COUNT(v.state_number) as cnt

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` v

            where v.hit_and_run='Yes'

            GROUP BY v.state_number) a

           inner join 

             (select distinct state_name, state_number 

             from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` ) sc 

        on a.state_number = sc.state_number

        order by cnt



        """

state_har = accidents.query_to_pandas_safe(query)

state_har

plt.xticks(rotation='vertical')

plt.bar(state_har.state_name, state_har.cnt)

plt.title("Number of Hit and Runs by State")


