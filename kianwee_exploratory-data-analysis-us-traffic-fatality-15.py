# Importing packages

import bq_helper as bqh # Simplifies common read-only tasks in BigQuery by dealing with object references and unpacking result objects into pandas dataframes

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
traffic_data = bqh.BigQueryHelper('bigquery-public-data','nhtsa_traffic_fatalities')
traffic_data.list_tables()
query = '''

SELECT state_name,

consecutive_number as case_id,

CASE

WHEN day_of_week = 1 THEN "Sunday"

WHEN day_of_week = 2 THEN "Monday"

WHEN day_of_week = 3 THEN "Tuesday"

WHEN day_of_week = 4 THEN "Wednesday"

WHEN day_of_week = 5 THEN "Thursday"

WHEN day_of_week = 6 THEN "Friday"

WHEN day_of_week = 7 THEN "Saturday"

ELSE "Unknown"

END day_of_week,

CASE

WHEN month_of_crash = 1 THEN 'Jan'

WHEN month_of_crash = 2 THEN 'Feb'

WHEN month_of_crash = 3 THEN 'Mar'

WHEN month_of_crash = 4 THEN 'Apr'

WHEN month_of_crash = 5 THEN 'May'

WHEN month_of_crash = 6 THEN 'Jun'

WHEN month_of_crash = 7 THEN 'Jul'

WHEN month_of_crash = 8 THEN 'Aug'

WHEN month_of_crash = 9 THEN 'Sep'

WHEN month_of_crash = 10 THEN 'Oct'

WHEN month_of_crash = 11 THEN 'Nov'

WHEN month_of_crash = 12 THEN 'Dec'

ELSE "Unknown"

END month_of_year,

CASE

WHEN hour_of_crash = 0 THEN '12AM'

WHEN hour_of_crash = 1 THEN '1AM'

WHEN hour_of_crash = 2 THEN '2AM'

WHEN hour_of_crash = 3 THEN '3AM'

WHEN hour_of_crash = 4 THEN '4AM'

WHEN hour_of_crash = 5 THEN '5AM'

WHEN hour_of_crash = 6 THEN '6AM'

WHEN hour_of_crash = 7 THEN '7AM'

WHEN hour_of_crash = 8 THEN '8AM'

WHEN hour_of_crash = 9 THEN '9AM'

WHEN hour_of_crash = 10 THEN '10AM'

WHEN hour_of_crash = 11 THEN '11AM'

WHEN hour_of_crash = 12 THEN '12PM'

WHEN hour_of_crash = 13 THEN '1PM'

WHEN hour_of_crash = 14 THEN '2PM'

WHEN hour_of_crash = 15 THEN '3PM'

WHEN hour_of_crash = 16 THEN '4PM'

WHEN hour_of_crash = 17 THEN '5PM'

WHEN hour_of_crash = 18 THEN '6PM'

WHEN hour_of_crash = 19 THEN '7PM'

WHEN hour_of_crash = 20 THEN '8PM'

WHEN hour_of_crash = 21 THEN '9PM'

WHEN hour_of_crash = 22 THEN '10PM'

WHEN hour_of_crash = 23 THEN '11PM'

ELSE "Unknown"

END hour_of_day,

number_of_fatalities as no_of_victims,

light_condition_name AS light_condition,

atmospheric_conditions_1_name AS atmospheric_condition,

from 

`bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

'''



traffic_data.estimate_query_size(query)
df = traffic_data.query_to_pandas(query)

df.head()
fatality_per_state = df.loc[:, ['state_name', 'no_of_victims']]

fatality_per_state = fatality_per_state.groupby(['state_name'])['no_of_victims'].sum().reset_index().sort_values('no_of_victims')



# Plotting using Seaborn

plt.figure(figsize=(14,5))

g = sns.barplot(x= "state_name", y = "no_of_victims", data = fatality_per_state)

state_name_list = pd.Series.tolist(fatality_per_state['state_name'])

y_pos = np.arange(len(state_name_list))

plt.xticks(y_pos, state_name_list, rotation='vertical')

plt.xlabel('State')

plt.ylabel('Fatality count')

g.axes.set_title('Graph showing number of fatality per state',fontsize=20)
fatality_per_week = df.loc[:, ['day_of_week', 'no_of_victims']]

fatality_per_week = fatality_per_week.groupby(['day_of_week'])['no_of_victims'].sum().reset_index().sort_values("no_of_victims")



# Plotting using Seaborn

plt.figure(figsize=(14,5))

g = sns.barplot(x= "day_of_week", y = "no_of_victims", data = fatality_per_week)

week_name_list = pd.Series.tolist(fatality_per_week['day_of_week'])

y_pos = np.arange(len(week_name_list))

plt.xticks(y_pos, week_name_list, rotation='vertical')

plt.xlabel('Day')

plt.ylabel('Fatality count')

g.axes.set_title('Graph showing number of fatality per day of week',fontsize=20)
fatality_per_day = df.loc[:, ['hour_of_day', 'no_of_victims']]

fatality_per_day = fatality_per_day.groupby(['hour_of_day'])['no_of_victims'].sum().reset_index().sort_values("no_of_victims")



# Plotting using Seaborn

plt.figure(figsize=(14,5))

g = sns.barplot(x= "hour_of_day", y = "no_of_victims", data = fatality_per_day)

day_name_list = pd.Series.tolist(fatality_per_day['hour_of_day'])

y_pos = np.arange(len(day_name_list))

plt.xticks(y_pos, day_name_list, rotation='vertical')

plt.xlabel('Hour')

plt.ylabel('Fatality count')

g.axes.set_title('Graph showing number of fatality per hour of day',fontsize=20)
fatality_per_year = df.loc[:, ['month_of_year', 'no_of_victims']]

fatality_per_year = fatality_per_year.groupby(['month_of_year'])['no_of_victims'].sum().reset_index().sort_values("no_of_victims")



# Plotting using Seaborn

plt.figure(figsize=(14,5))

g = sns.barplot(x= "month_of_year", y = "no_of_victims", data = fatality_per_year)

year_name_list = pd.Series.tolist(fatality_per_year['month_of_year'])

y_pos = np.arange(len(year_name_list))

plt.xticks(y_pos, year_name_list, rotation='vertical')

plt.xlabel('Month')

plt.ylabel('Fatality count')

g.axes.set_title('Graph showing number of fatality per month of year',fontsize=20)
fatality_light_condition = df.loc[:, ['light_condition', 'no_of_victims']]

fatality_light_condition = fatality_light_condition.groupby(['light_condition'])['no_of_victims'].sum().reset_index().sort_values("no_of_victims")



# Plotting using Seaborn

plt.figure(figsize=(14,5))

g = sns.barplot(x= "light_condition", y = "no_of_victims", data = fatality_light_condition)

light_cond = pd.Series.tolist(fatality_light_condition['light_condition'])

y_pos = np.arange(len(light_cond))

plt.xticks(y_pos, light_cond, rotation='vertical')

plt.xlabel('Light condition')

plt.ylabel('Fatality count')

g.axes.set_title('Graph showing number of fatality per light condition',fontsize=20)
fatality_atmospheric_condition = df.loc[:, ['atmospheric_condition', 'no_of_victims']]

fatality_atmospheric_condition = fatality_atmospheric_condition.groupby(['atmospheric_condition'])['no_of_victims'].sum().reset_index().sort_values("no_of_victims")



# Plotting using Seaborn

plt.figure(figsize=(14,5))

g = sns.barplot(x= "atmospheric_condition", y = "no_of_victims", data = fatality_atmospheric_condition)

atmospheric_cond = pd.Series.tolist(fatality_atmospheric_condition['atmospheric_condition'])

y_pos = np.arange(len(atmospheric_cond))

plt.xticks(y_pos, atmospheric_cond, rotation='vertical')

plt.xlabel('Atmospheric condition')

plt.ylabel('Fatality count')

g.axes.set_title('Graph showing number of fatality per atmospheric condition',fontsize=20)