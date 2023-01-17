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

# import package with helper functions 

import bq_helper



# create a helper object for this dataset

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="nhtsa_traffic_fatalities")

#get all tables in this dataset

accidents.list_tables()

# heg 5 rows of 'accident_2015' tale

accidents.head('accident_2015')



#query to find which day of the week the most fatal traffic accidents happen on.

accidental_day_of_week_query = """SELECT COUNT(consecutive_number) as count_id, EXTRACT (DAYOFWEEK FROM timestamp_of_crash) as day_week

                                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

                                    

                                    GROUP BY day_week

                                    ORDER BY count_id DESC"""



#quwry to pandas

accidental_day_of_week = accidents.query_to_pandas_safe(accidental_day_of_week_query)

# see table result

accidental_day_of_week[['count_id','day_week']]

# or see result

print(accidental_day_of_week)
# library for plotting

import matplotlib.pyplot as plt



# make a plot to show that our data is, actually, sorted:

plt.plot(accidental_day_of_week.count_id)

plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
# query to find hours of the day do the most accidents accur during

accidental_hour_day_query = """ SELECT COUNT(consecutive_number) as countID ,EXTRACT (HOUR FROM timestamp_of_crash) as hour_day

                         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

                         GROUP BY hour_day

                         ORDER BY countID DESC"""



# convert query to pandas table 

accidental_hour_day = accidents.query_to_pandas_safe(accidental_hour_day_query)



#print result

print(accidental_hour_day)



# more practice for myself

plt.plot(accidental_hour_day.hour_day)
# query to find which state has the most hit and runs

hit_and_run_state_query = """ SELECT COUNT(hit_and_run) as count_hit_run, registration_state_name

                              FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`                        

                              GROUP by registration_state_name

                              ORDER by count_hit_run DESC"""



# query to pandas 

hit_and_run_state = accidents.query_to_pandas_safe(hit_and_run_state_query)



# see result

print(hit_and_run_state)