# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

accidents = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                    dataset_name = "nhtsa_traffic_fatalities")
#accidents.list_tables()
#accidents.table_schema('accident_2015')
query = """SELECT COUNT(consecutive_number),
                EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT (DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number)
        """

accidents.estimate_query_size(query)
acc_by_dayofweek = accidents.query_to_pandas_safe(query)
acc_by_dayofweek.sort_values(by=['f1_'], ascending = False, inplace = True)
acc_by_dayofweek.reset_index(inplace=True)
acc_by_dayofweek
import matplotlib.pyplot as plt

plt.plot(acc_by_dayofweek.f0_)
plt.show()
query2 = """SELECT COUNT(consecutive_number) AS num_acc,
                    EXTRACT (HOUR FROM timestamp_of_crash) AS hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT (HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
            """

accidents.estimate_query_size(query)
acc_by_hour = accidents.query_to_pandas_safe(query2)
acc_by_hour
acc_by_hour_ord = acc_by_hour.sort_values(by=['f1_'], inplace =True)
print(acc_by_hour)
#plt.plot(acc_by_hour.f0_)
plt.scatter(acc_by_hour.f1_, acc_by_hour.f0_)
plt.show()
x = np.arange(24)
plt.bar(x,acc_by_hour.f0_)
plt.show()

accidents.head('vehicle_2015')
query3 = """SELECT COUNT(hit_and_run) AS hit_and_runs,
                    state_number AS state
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY state_number
            ORDER BY COUNT(hit_and_run)
            """

accidents.estimate_query_size(query3)
har_by_st = accidents.query_to_pandas_safe(query3)
har_by_st.head()
har_by_st.sort_values(by=['state'],inplace=True)
har_by_st.head()
len(har_by_st)
#x_st = np.arange(51)
#plt.plot(har_by_st.hit_and_runs)
plt.scatter(har_by_st.state, har_by_st.hit_and_runs)
query4 = """SELECT COUNT(hit_and_run) AS hit_and_runs,
                    registration_state_name AS state
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
            """

accidents.estimate_query_size(query4)
acc_by_st_name = accidents.query_to_pandas_safe(query4)
acc_by_st_name.head(10)
