# general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the helper class
import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                    dataset_name="nhtsa_traffic_fatalities")
# Show a list of tables
accidents.list_tables()
# Examine one of the tables
accidents.head("accident_2016")
# build our query
query1 = """
        SELECT COUNT(consecutive_number) as count, EXTRACT(HOUR FROM timestamp_of_crash) as hour
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        GROUP BY hour
        ORDER BY count DESC
"""

# check how much data query will scan
accidents.estimate_query_size(query1)

# run query and store results in dataframe
crashtime = accidents.query_to_pandas_safe(query1)
# reorder columns so that the hour is first (easier to read)
crashtime = crashtime[['hour', 'count']]

# show the results
crashtime
sorted = crashtime.sort_values('hour')

%matplotlib inline
plt.style.use('ggplot')
plt.figure(figsize=(16,6))
plt.bar(range(len(sorted)), sorted['count'], color='blue')
plt.xticks(range(24))
plt.xlabel('Hour of Day (24-hour)')
plt.ylabel('No. of Crashes')
# Build our query
query2 = """
        SELECT COUNT(hit_and_run) as hit_and_runs, registration_state_name as state
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
        WHERE hit_and_run = 'Yes'
        GROUP BY state
        ORDER BY hit_and_runs DESC
"""

# see how much data the query will use
accidents.estimate_query_size(query2)
# run the query and save to dataframe
hit_and_runs = accidents.query_to_pandas_safe(query2)
hit_and_runs
# Organize, exclude "Unknown" and shorten the two very long state descriptions for graphing purposes
sorted = hit_and_runs.sort_values('hit_and_runs')
sorted = sorted[sorted['state']!='Unknown']
sorted.loc[sorted['state'] == 'Other Registration (Includes Native American Indian Nations)', 'state'] = 'Other Registration'
sorted.loc[sorted['state'] == 'U.S. Government Tags (Includes Military)', 'state'] = 'U.S. Govt. Tags'
states = sorted['state']
hitandruns = sorted['hit_and_runs']

fig, ax = plt.subplots()
fig.set_size_inches(12,12)
ax.barh(np.arange(len(states)), hitandruns, color='dodgerblue', align='edge')
ax.set_yticklabels(states)
ax.set_yticks(np.arange(len(states)))
ax.set_xlabel("No. of Hit and Runs")
ax.set_title("Hit and Runs in 2016 (Ranked by State of Car Registration)*")
fig.text(0.5,.05, "* In 2016, there were 929 hit and runs involving cars with unknown states of registration.", ha='right')
plt.margins(y=0.01)
