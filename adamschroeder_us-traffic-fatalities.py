import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head('accident_2015')
query = """SELECT state_name, COUNT(consecutive_number) AS drunk_accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            WHERE number_of_drunk_drivers > 0
            GROUP BY state_name
            ORDER BY drunk_accidents DESC 
        """
# accidents.estimate_query_size(query)
drunk_accident = accidents.query_to_pandas_safe(query, max_gb_scanned=0.5)
query1 = """SELECT state_name, COUNT(consecutive_number) AS all_accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY state_name
            ORDER BY all_accidents DESC 
        """
# accidents.estimate_query_size(query1)
all_accident = accidents.query_to_pandas_safe(query1, max_gb_scanned=0.1)
# Merge datasets
state_accident = pd.merge(all_accident, drunk_accident, on='state_name')

# caculate percentage of alcohol-related accidents
state_accident['prcnt_drunk'] = state_accident['drunk_accidents'] / state_accident['all_accidents']
sortBydrunkPrcnt = state_accident.sort_values(by='prcnt_drunk', ascending=False)
fig=plt.gcf()
fig.set_size_inches(14,6)
sns.barplot('state_name', 'prcnt_drunk', data=sortBydrunkPrcnt)
plt.xticks(rotation=90)
plt.title("Percentage of accidents that involve drunk drivers")
plt.show()
query2 = """SELECT COUNT(consecutive_number) AS num_accidents,
                   EXTRACT(HOUR FROM timestamp_of_crash) AS hour_of_crash
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            WHERE number_of_drunk_drivers = 0   --accidents without alcohol
            GROUP BY hour_of_crash
            ORDER BY num_accidents DESC
        """
# accidents.estimate_query_size(query2)
accidentsNOalcohol = accidents.query_to_pandas_safe(query2, max_gb_scanned=0.01)
query3 = """SELECT COUNT(consecutive_number) AS num_accidents,
                   EXTRACT(HOUR FROM timestamp_of_crash) AS hour_of_crash
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            WHERE number_of_drunk_drivers > 0   --accidents without alcohol
            GROUP BY hour_of_crash
            ORDER BY num_accidents DESC
        """
# accidents.estimate_query_size(query3)
accidentsYESalcohol = accidents.query_to_pandas_safe(query3, max_gb_scanned=0.01)
fig=plt.gcf()
fig.set_size_inches(10,5)
sns.barplot('hour_of_crash', 'num_accidents', data=accidentsYESalcohol)
plt.title("Amount/hour of fatal accidents involving alcohol")
plt.show()

fig=plt.gcf()
fig.set_size_inches(10,5)
sns.barplot('hour_of_crash', 'num_accidents', data=accidentsNOalcohol)
plt.title("Amount/hour of fatal accidents without alcohol")
plt.show()
query4 = """SELECT atmospheric_conditions_1_name, COUNT(DISTINCT consecutive_number) AS num_of_accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY atmospheric_conditions_1_name
            ORDER BY num_of_accidents DESC
         """
accidents.estimate_query_size(query4)
atmospheric_accidents = accidents.query_to_pandas_safe(query4, max_gb_scanned=0.01)
first_five = atmospheric_accidents[0:5]
atmospheric_labels = first_five.atmospheric_conditions_1_name
plt.figure(figsize=(8,5))
plt.pie(first_five.num_of_accidents, labels=atmospheric_labels, autopct='%1.1f%%')
plt.show()
query5 = """SELECT state_name, consecutive_number, hour_of_notification, minute_of_notification,
                   hour_of_arrival_at_scene, minute_of_arrival_at_scene
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            """
accidents.estimate_query_size(query5)
arrival_of_ems = accidents.query_to_pandas_safe(query5, max_gb_scanned=0.01)
# remove rows with hours over 23 and minutes over 60
f = arrival_of_ems[(arrival_of_ems['hour_of_notification']>=0 )&\
                   (arrival_of_ems['hour_of_notification']<24) &\
                   (arrival_of_ems['minute_of_notification']>=0)&\
                   (arrival_of_ems['minute_of_notification']<60)]
f = f[(f['hour_of_arrival_at_scene']>=0 )&\
      (f['hour_of_arrival_at_scene']<24)&\
      (f['minute_of_arrival_at_scene']>=0)&\
      (f['minute_of_arrival_at_scene']<60)]
f['next_day'] = f.hour_of_arrival_at_scene - f.hour_of_notification

f.hour_of_notification = f.hour_of_notification.apply(lambda x: str(x))
f.minute_of_notification = f.minute_of_notification.apply(lambda x: str(x))
f['notification'] = f.hour_of_notification+':'+f.minute_of_notification
f['notification'] = f['notification'].apply(lambda x: pd.to_datetime(x, format='%H:%M'))

f.hour_of_arrival_at_scene = f.hour_of_arrival_at_scene.apply(lambda x: str(x))
f.minute_of_arrival_at_scene = f.minute_of_arrival_at_scene.apply(lambda x: str(x))
f['arrival'] = f.hour_of_arrival_at_scene+':'+f.minute_of_arrival_at_scene
f['arrival'] = f['arrival'].apply(lambda x: pd.to_datetime(x, format='%H:%M'))

f['arrival_time'] = f.apply(lambda row: row['arrival'].replace(day=2) if row['next_day']<0 else row['arrival'],  axis=1)
f.drop(labels='arrival', inplace=True, axis=1)
f['time_dfrc'] = f['arrival_time']-f['notification']
f['time_dfrc'] = f['time_dfrc'].apply(lambda x: pd.Timedelta(x).total_seconds() / 60)

plt.figure(figsize=(12,4))
f.groupby('state_name')['time_dfrc'].mean().sort_values().plot.bar()
plt.title("Minutes (on average) it takes emergency medical services to arrive on scene")
plt.ylabel("Average Minutes")
plt.show()
# f.groupby('state_name')['time_dfrc'].mean().sort_values()
query6 = """SELECT COUNT(DISTINCT consecutive_number) AS num_fatal_accidents, 
                    vehicle_identification_number_vin AS vehicle_ID
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY vehicle_ID
            ORDER BY num_fatal_accidents DESC
            """
accidents.estimate_query_size(query6)
cursed_vehicles = accidents.query_to_pandas_safe(query6, max_gb_scanned=0.01)
plt.figure(figsize=(10,4))
plt.plot(cursed_vehicles[3:21].reset_index().num_fatal_accidents, 'ro')
labels=cursed_vehicles[3:21].reset_index().vehicle_ID
plt.xticks(range(len(cursed_vehicles[3:21])), labels)
plt.xticks(rotation=90)
plt.xlabel("Vin Number")
plt.ylabel("Num_of_Fatal_Accidents")
plt.title("Vehicles involved in 5 or more fatal accidents")
plt.show()
query7 = """SELECT DISTINCT (consecutive_number) AS accident_num, alcohol_test_status3, 
                    person_type_name, alcohol_test_status1, sex, age
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.person_2015`
            WHERE alcohol_test_status1='Test Given'
            """
accidents.estimate_query_size(query7)
g = accidents.query_to_pandas_safe(query7, max_gb_scanned=0.01)
# clean data
g.alcohol_test_status3 = g.alcohol_test_status3.astype(int)
g = g[(g['alcohol_test_status3']<941)& (g['alcohol_test_status3']>0)]
g = g[g['age']<800]
g = g[(g['sex']!='Not Reported') & (g['sex']!='Unknown') ]
g = g[(g['person_type_name']=='Driver of a Motor Vehicle In-Transport') |
      (g['person_type_name']=='Passenger of a Motor Vehicle In-Transport') | 
      (g['person_type_name']=='Pedestrian') |
      (g['person_type_name']=='Bicyclist')]
g.alcohol_test_status3.plot.hist(bins=14)
plt.title("Alcohol levels by BAC test")
plt.xlabel("Blood Alcohol Percentage")
plt.xlim(0,600)
plt.show()
print("Illegal intoxication levels in most states is over 80")
g['age_range'] = 0
mask = (g['age'] >= 0) & (g['age'] < 21)
g.loc[mask, 'age_range'] = '0-20'
mask = (g['age'] > 20) & (g['age'] < 31)
g.loc[mask, 'age_range'] = '21-30'
mask = (g['age'] > 30) & (g['age'] < 41)
g.loc[mask, 'age_range'] = '31-40'
mask = (g['age'] > 40) & (g['age'] < 61)
g.loc[mask, 'age_range'] = '41-60'
mask = (g['age'] > 60) & (g['age'] < 100)
g.loc[mask, 'age_range'] = '61-100'
sns.countplot(x='sex', data=g)
fig=plt.gcf()
fig.set_size_inches(6,4)
plt.show()
sns.countplot(x="age_range", hue="sex", data=g)
plt.title("Men/Women tested for alcohol")
plt.show()
sns.pointplot('age_range', 'alcohol_test_status3', data=g, estimator=np.mean, hue='sex', dodge=True,\
              linestyles=["-", "--"], markers=["x", "o"])
fig=plt.gcf()
fig.set_size_inches(14,4)
plt.ylabel("Blood Alcohol Percentage")
plt.title("Average alcohol found in blood by gender and age")
plt.show()