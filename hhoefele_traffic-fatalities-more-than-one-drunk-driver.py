# import pandas
import pandas as pd

# import Kaggle's bq_helper package
import bq_helper
# create a helper object for this bigquery dataset
traffic_fatalities = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                             dataset_name="nhtsa_traffic_fatalities" )
# query to calculate number of accidents, fatalities, vehicles, drunk drivers (number and percent)
# grouped by state, sorted by number of accidents

# 2015

query = """SELECT
  COUNT(consecutive_number) AS accidents_2015,
  SUM(number_of_fatalities) AS fatalities_2015,
  SUM(number_of_vehicle_forms_submitted_all) AS vehicles_2015,
  SUM(number_of_drunk_drivers) AS drunk_drivers_2015,
  (SUM(number_of_drunk_drivers) / COUNT(consecutive_number))*100 AS pct_drunk_drivers_2015
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
ORDER BY
  accidents_2015 DESC
"""

#2016

query2 = """SELECT
  COUNT(consecutive_number) AS accidents_2016,
  SUM(number_of_fatalities) AS fatalities_2016,
  SUM(number_of_vehicle_forms_submitted_all) AS vehicles_2016,
  SUM(number_of_drunk_drivers) AS drunk_drivers_2016,
 (SUM(number_of_drunk_drivers) / COUNT(consecutive_number))*100 AS pct_drunk_drivers_2016
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
ORDER BY
  accidents_2016 DESC
"""
# run query and store in dataframe if safe size (less than 1 GB) , or add parameter (...,max_gb_scanned = 1)
fatalities_2015 = traffic_fatalities.query_to_pandas_safe(query)
fatalities_2016 = traffic_fatalities.query_to_pandas_safe(query2)
# check query size
traffic_fatalities.estimate_query_size(query)
traffic_fatalities.estimate_query_size(query2)
fatalities_2015
fatalities_2016
# query to calculate number of accidents, fatalities, vehicles, drunk drivers (number and percent)
# grouped by state, sorted by number of accidents

# 2015
query3 = """SELECT state_name,
  COUNT(consecutive_number) AS accidents_2015,
  SUM(number_of_fatalities) AS fatalities_2015,
  SUM(number_of_vehicle_forms_submitted_all) AS vehicles_2015,
  SUM(number_of_drunk_drivers) AS drunk_drivers_2015,
  (SUM(number_of_drunk_drivers) / COUNT(consecutive_number))*100 AS pct_drunk_drivers_2015
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY state_name
ORDER BY
  accidents_2015 DESC
"""

#2016

query4 = """SELECT state_name,
  COUNT(consecutive_number) AS accidents_2016,
  SUM(number_of_fatalities) AS fatalities_2016,
  SUM(number_of_vehicle_forms_submitted_all) AS vehicles_2016,
  SUM(number_of_drunk_drivers) AS drunk_drivers_2016,
 (SUM(number_of_drunk_drivers) / COUNT(consecutive_number))*100 AS pct_drunk_drivers_2016
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY  state_name
ORDER BY
  accidents_2016 DESC
"""
# run query and store in dataframe if safe size (less than 1 GB) , or add parameter (...,max_gb_scanned = 1)
fatalities_by_state_2015 = traffic_fatalities.query_to_pandas_safe(query3)
fatalities_by_state_2016 = traffic_fatalities.query_to_pandas_safe(query4)
# check query size
traffic_fatalities.estimate_query_size(query3)
traffic_fatalities.estimate_query_size(query4)
fatalities_by_state_2015.head(5)
fatalities_by_state_2016.head(5)
# save dataframe as a csv file
fatalities_by_state_2015.to_csv("fatalities_2015")
fatalities_by_state_2016.to_csv("fatalities_2016")
# https://www.kaggle.com/crawford/python-merge-tutorial
# Merging dataframes
# Outer merge on ID column
# pd.merge(left=left_dataframe, right=right_dataframe, on="ID", how="outer")
fatalities_by_state_2015_2016 = pd.merge(left=fatalities_by_state_2015, right= fatalities_by_state_2016, on="state_name", how="outer")
fatalities_by_state_2015_2016.head()
# save merged dataframe to csv file
fatalities_by_state_2015_2016.to_csv("fatalities_by_state_2015_2016")
# query to find out the number of accidents which happen by day of week in 2016

query5 = """SELECT
            EXTRACT(DAYOFWEEK FROM timestamp_of_crash) day_of_week,
            COUNT(consecutive_number) accidents_count
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY day_of_week
            ORDER BY day_of_week
        """

# check query size
traffic_fatalities.estimate_query_size(query5)
# run query and store in dataframe if safe size (less than 1 GB) , or add parameter (...,max_gb_scanned = 1)
fatalities_by_day_2016 = traffic_fatalities.query_to_pandas_safe(query5)
# print first couple lines of the "fatalities_by_week_2015" table
fatalities_by_day_2016
# save dataframe as a csv file
fatalities_by_day_2016.to_csv("fatalities_by_dayofweek_2016")
#library for plotting
import matplotlib.pyplot as plt
# plot number accidents per month 2016
plt.plot(fatalities_by_day_2016.accidents_count)
# query to find out the number of accidents in state where they occurred (not regisration state) 
# that involved one or more drunk driver, sorted by count of drunk driver incidents.
query6 = """SELECT state_name, COUNT(consecutive_number) accidents_count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY state_name, number_of_drunk_drivers
            HAVING number_of_drunk_drivers > 0
            ORDER BY accidents_count DESC
        """

# check query size
traffic_fatalities.estimate_query_size(query6)
# run query and store in dataframe if safe size (less than 1 GB) , or add parameter (...,max_gb_scanned = 1)
drunk_incidents_by_state_2016 = traffic_fatalities.query_to_pandas_safe(query6)
# print first couple lines of the "drunk_incidents_by_state_2015" table
drunk_incidents_by_state_2016.head()
# with help from: https://stackoverflow.com/questions/2360396/how-can-i-merge-the-columns-from-two-tables-into-one-output

query7 = """

SELECT T1.state_name, accidents_2015, accidents_2016, 
                ((accidents_2016-accidents_2015)/accidents_2015)*100 pct_chg, 
                drunk_drivers_2015, drunk_drivers_2016,
                pct_drunk_drivers_2015, pct_drunk_drivers_2016, 
                pct_drunk_drivers_2016-pct_drunk_drivers_2015 pct_drunk_drivers_chg
FROM

(SELECT state_name ,
  COUNT(consecutive_number) AS accidents_2015,
  SUM(number_of_fatalities) AS fatalities_2015,
  SUM(number_of_vehicle_forms_submitted_all) AS vehicles_2015,
  SUM(number_of_drunk_drivers) AS drunk_drivers_2015,
  (SUM(number_of_drunk_drivers) / COUNT(consecutive_number))*100 AS pct_drunk_drivers_2015
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY
  state_name)  T1

INNER JOIN
  
(SELECT state_name ,
  COUNT(consecutive_number) AS accidents_2016,
  SUM(number_of_fatalities) AS fatalities_2016,
  SUM(number_of_vehicle_forms_submitted_all) AS vehicles_2016,
  SUM(number_of_drunk_drivers) AS drunk_drivers_2016,
  (SUM(number_of_drunk_drivers) / COUNT(consecutive_number))*100 AS pct_drunk_drivers_2016
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY
  state_name)   T2
  
ON T1.state_name = T2.state_name

ORDER BY accidents_2016 DESC

  
"""

# check query size
traffic_fatalities.estimate_query_size(query7)
# run query and store in dataframe if safe size (less than 1 GB) , or add parameter (...,max_gb_scanned = 1)
fatalities_by_state_all = traffic_fatalities.query_to_pandas_safe(query7)
fatalities_by_state_all.head(5)
# https://community.modeanalytics.com/sql/tutorial/sql-case/
query8 = """
SELECT CASE WHEN number_of_drunk_drivers = 0 THEN 'zero'
            WHEN number_of_drunk_drivers = 1 THEN 'one'
            WHEN number_of_drunk_drivers = 2 THEN 'two'
            WHEN number_of_drunk_drivers = 3 THEN 'three'
            WHEN number_of_drunk_drivers > 3 THEN 'more_than_three'
            ELSE 'other' END AS number_of_drunk_drivers_group,
            COUNT(1) AS count
  FROM  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
 GROUP BY 1
"""

# check query size
traffic_fatalities.estimate_query_size(query8)
# run query and store in dataframe if safe size (less than 1 GB) , or add parameter (...,max_gb_scanned = 1)
fatalities_by_state_drunkdriver_groups = traffic_fatalities.query_to_pandas_safe(query8)
fatalities_by_state_drunkdriver_groups.head(10)
query9 = """
SELECT state_name, number_of_fatalities, number_of_drunk_drivers,timestamp_of_crash,
   number_of_vehicle_forms_submitted_all,number_of_persons_in_motor_vehicles_in_transport_mvit
   FROM  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
   WHERE number_of_drunk_drivers >= 3 
 ORDER BY 2
"""

# check query size
traffic_fatalities.estimate_query_size(query9)
# run query and store in dataframe if safe size (less than 1 GB) , or add parameter (...,max_gb_scanned = 1)
incidents_more_than_3_drunks_drivers = traffic_fatalities.query_to_pandas_safe(query9)
incidents_more_than_3_drunks_drivers.head(10)
# What is the maximum number of fatalities in one incident in 2016
query77 = """  SELECT year_of_crash, max(number_of_fatalities) AS max_fatalities
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY year_of_crash
        """
# check query size
traffic_fatalities.estimate_query_size(query77)
# run query and store in dataframe if safe size (less than 1 GB) , or add parameter (...,max_gb_scanned = 1)
max_query = traffic_fatalities.query_to_pandas_safe(query77)
max_query.head(10)
# SQL Max Function Examples: https://www.techonthenet.com/sql/max.php
# Which incident involved the most fatalities in 2016?
query10 = """  
          SELECT ht.Consecutive_Number, ht.state_name, ht.trafficway_identifier,
            ht.timestamp_of_crash, ht.number_of_fatalities, ht.manner_of_collision_name,
            ht.number_of_drunk_drivers,
            ht.number_of_vehicle_forms_submitted_all,
            ht.first_harmful_event_name,
            ht.number_of_persons_in_motor_vehicles_in_transport_mvit
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` ht,
            
            (SELECT year_of_crash, max(number_of_fatalities) AS max_fatalities
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY year_of_crash) maxid
            
            WHERE  ht.number_of_fatalities = maxid.max_fatalities
        """
# check query size
traffic_fatalities.estimate_query_size(query10)
# run query and store in dataframe if safe size (less than 1 GB) , or add parameter (...,max_gb_scanned = 1)
incident_with_most_fatalities = traffic_fatalities.query_to_pandas_safe(query10)
incident_with_most_fatalities.head(10)