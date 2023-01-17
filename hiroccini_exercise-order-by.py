import pandas as pd
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# Your Code Here
print(f"{accidents.list_tables()}")
accidents.head("accident_2015")
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
query_2015 = """SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
ORDER BY COUNT(consecutive_number) DESC"""
accidents_by_hour_2015 = accidents.query_to_pandas_safe(query_2015)
query_2016 = """SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
ORDER BY COUNT(consecutive_number) DESC"""
accidents_by_hour_2016 = accidents.query_to_pandas_safe(query_2016)
accidents_by_hour_2015.head()
accidents_by_hour_2016.head()
l_2015 = [n for n in accidents_by_hour_2015['f1_']]
l_2016 = [n for n in accidents_by_hour_2016['f1_']]

print(f"{l_2015}\n{l_2016}")
print(f"{sorted(l_2015)}\n{sorted(l_2016)}")
print(f"{[l for l in l_2015 if l not in l_2016]}")
# for v in accidents_by_hour_2015.values:
#     print(v[0], v[1])
print({v[0]: v[1] for v in accidents_by_hour_2015.values})
print({v[0]: v[1] for v in accidents_by_hour_2016.values})
print("")
# acc_merge = pd.merge(accidents_by_hour_2015, accidents_by_hour_2016, on='f1_')
acc_merge = pd.merge(accidents_by_hour_2015, accidents_by_hour_2016, on='f1_', how='inner',
                    suffixes=['2015', '2016']) # on='f1_で'pd.merge() how='inner' はなくても可
# print(f"{acc_merge.head()}")
acc_merge_index = acc_merge.set_index('f1_')
print(f"{acc_merge_index.head()}")
fig, axes = plt.subplots(figsize=(14,2))
acc_merge_index.plot.bar(ax=axes)
for col in acc_merge_index.columns:
    print(f"{col[-4:]}: {acc_merge_index[col].mean()}")
# {"average_{}".format(col[-4:]): acc_merge_index[col].mean() for col in acc_merge_index.columns}
# Your Code Here
accidents.head("vehicle_2015")
# # for Check
# query_hr_2015 = """SELECT hit_and_run
# FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
# GROUP BY hit_and_run"""

query_hr_2015 = """SELECT state_number, hit_and_run, count(*)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
GROUP BY state_number, hit_and_run """

# # for pd.pivot_table
# query_hr_2015 = """SELECT state_number, hit_and_run
# FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`"""

hit_and_run_2015 = accidents.query_to_pandas_safe(query_hr_2015)
print(f"{hit_and_run_2015.shape}")
print(f"{hit_and_run_2015.head()}")
hit_and_run_2015[hit_and_run_2015.state_number == 1]
yes_row = hit_and_run_2015['hit_and_run'] == 'Yes'
answer_2 = hit_and_run_2015.loc[yes_row, ['state_number', 'f0_']].sort_values('f0_', 
                                                                              ascending=False)
print(f"{answer_2.head()}")
fig,axes = plt.subplots(figsize=(12,2))
answer_2.set_index('state_number').plot.bar(ax=axes)
# for col in hit_and_run_2015.columns:
#     print(f"{col}: {hit_and_run_2015[col].unique()}")
df_pivot = pd.pivot_table(hit_and_run_2015, index=['state_number'], columns=['hit_and_run'], 
               aggfunc=lambda x:x).fillna(0).astype(int)
print(f"{df_pivot.columns}\n{df_pivot.head()}\n")
df_pivot_sort = df_pivot['f0_'].loc[:, ['Yes', 'No', 'Unknown']]\
                .sort_values(by='Yes', ascending=False)  # ['f0_'] がミソ
print(f"{df_pivot_sort.columns}\n{df_pivot_sort.head()}")