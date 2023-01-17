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

accidents = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                    dataset_name='nhtsa_traffic_fatalities')

query = """
select
  extract(year from timestamp_of_crash) as year
, extract(hour from timestamp_of_crash) as hour
, count(*) as count_accidents
from
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
group by
  extract(year from timestamp_of_crash)
, extract(hour from timestamp_of_crash)
union all
select
  extract(year from timestamp_of_crash) as year
, extract(hour from timestamp_of_crash) as hour
, count(*) as count_accidents
from
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
group by
  extract(year from timestamp_of_crash)
, extract(hour from timestamp_of_crash)
"""

a = accidents.query_to_pandas_safe(query)
for y in a.year.unique():
    print('Five Most Dangerous Hours in {}'.format(y))
    print('---------------------------------')
    print(a[a.year==y].sort_values('count_accidents', ascending=False).head(5))
    print('\n')
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.barplot(
    x='hour',
    y='count_accidents',
    data=a.sort_values('count_accidents'),
    hue='year',
    ax=ax
)
ax.set_title('Accidents by Hour\n2015-2016')
ax.set_ylabel('Count of Accidents')
ax.set_xlabel('Hour of Day')
query = """
select
  registration_state_name
, count(*) count_registered_vehicles
from
  `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
where
  registered_vehicle_owner != 0
  and hit_and_run = 'Yes'
group by
  registration_state_name
order by
  count(*) desc
"""

hnr = accidents.query_to_pandas_safe(query)
print(hnr.head(5))
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(
    x='registration_state_name',
    y='count_registered_vehicles',
    data=hnr.head(10),
    ax=ax,
    palette='GnBu_d'
)
ax.set_title('Top 10 States for Hit and Run Accidents\n2016')
ax.set_ylabel('Count of Hit and Run Accidents')
ax.set_xlabel('State')