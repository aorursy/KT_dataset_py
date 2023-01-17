import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
# Which day of the week the most fatal traffic accidents happen on
query = """
select day_of_week, count(*)
from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
group by day_of_week
order by count(*) desc
"""
fatalities_by_day = accidents.query_to_pandas_safe(query)
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(fatalities_by_day.f0_)
# fatalities_by_day.sort(['day_of_week',], ascending)
# Error: 'DataFrame' object has no attribute 'sort'
# plt.plot(fatalities_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
# Which hours of the day do the most accidents occur during?
# Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour. Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
# Hint: You will probably want to use the EXTRACT() function for this.

query = """
select extract(hour from timestamp_of_crash) as hour, count(*)
from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
group by hour
"""
fatalities_by_hour = accidents.query_to_pandas_safe(query)
plt.plot(fatalities_by_hour.f0_)
# Which state has the most hit and runs?
# Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.

# hit_run_vehicles_per_state

query = """
select registration_state, count(*)
from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
where hit_and_run = 'Yes'
group by registration_state
order by count(*) desc
limit 10
"""

statewise_hit_and_run = accidents.query_to_pandas_safe(query)
statewise_hit_and_run