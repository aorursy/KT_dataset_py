# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# import bq_helper package with helper functions 
import bq_helper
# create a helper object for this dataset
traffic_fatality = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                  dataset_name = "nhtsa_traffic_fatalities")
# print all the tables in this dataset (there's only one!)
traffic_fatality.list_tables()

# Here we will be working with 2015 tables only
# print information on all the columns in the "global_air_quality" table
# in the OpenAQ dataset
traffic_fatality.table_schema("accident_2015")

# preview the first couple lines of the "accident_2015" table
traffic_fatality.head("accident_2015")

# Qeury to print out state-wsie accidents in 2015 in descending order(hishest to lowest)
# more than 1000 accidents in a year
query = """SELECT state_name,state_number,COUNT(consecutive_number) AS accident_counts
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY state_name,state_number
        HAVING COUNT(consecutive_number)>1000
        ORDER BY accident_counts DESC """

# Estimate query size before executing it
traffic_fatality.estimate_query_size(query)

# As the szie is very less, 0.79MB, running without safe mode
# store result to visualize in a bar-chart
state_accidents = traffic_fatality.query_to_pandas(query)

# library for plotting
import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')

# make a plot to show that our data is, actually, sorted:
plt.subplot(2, 1, 1)
plt.bar(state_accidents.state_name,state_accidents.accident_counts)
plt.title("2015: State-wise Accidents > 1000 - with average line")

# Print out an average line
plt.axhline(state_accidents.accident_counts.mean(), color='b', linestyle='dashed', linewidth=2)

# Print out top 10 states contributing 2015 fatalities
state_accidents.head(10)

# Find out which county/city-wise fatalities for each of the states
# We could have done this by HAVING COUNT()
# However to learn the usage we are going to use IN operator of SQL
query = """ SELECT state_name, county, city, MAX(accident_counts) AS max_accidents
              FROM (SELECT state_name, county, city, COUNT(consecutive_number) AS accident_counts
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                     WHERE state_name IN ("Texas","California","Florida","Goergia","North Calorina",
                                          "Pennsylvania","New York","Ohio","Illinois","South Carolina")
                     GROUP BY state_name, county, city
                    ) 
             GROUP BY state_name, county, city
             ORDER BY max_accidents DESC
        """

# Estimate query size before executing it
traffic_fatality.estimate_query_size(query)

# As the szie is very less, 1MB, running without safe mode
max_accidents = traffic_fatality.query_to_pandas(query)

# Print out top 10 county/city combination for the states under consideration
max_accidents.head(10)

# Lets figure out what all other factors involved in such high number of crashes
# we are considering only few on them at this moment
# The publication is not available to interpret the different codes used
# https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812315
# Hence we are not working with them as of now
query = """ SELECT state_name, month_of_crash,
                   hour_of_crash, national_highway_system,
                   light_condition, light_condition_name, school_bus_related,
                   rail_grade_crossing_identifier, number_of_fatalities,
                   number_of_drunk_drivers, COUNT(consecutive_number) AS accident_counts
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            WHERE state_name IN ("Texas","California","Florida","Goergia","North Calorina",
                                 "Pennsylvania","New York","Ohio","Illinois","South Carolina")
            GROUP BY state_name, month_of_crash,
                     hour_of_crash, national_highway_system,
                     light_condition, light_condition_name, school_bus_related,
                     rail_grade_crossing_identifier, number_of_fatalities,
                     number_of_drunk_drivers
            ORDER BY accident_counts DESC
        """

# Estimate query size
traffic_fatality.estimate_query_size(query)

# 0.0026095984503626823GB or 2.6MB of space required which is very minimal
# lets run without safe mode and store results in a dataframe
accident_factors = traffic_fatality.query_to_pandas(query)

# Print out few lines of records
accident_factors.head(5)

# Create a scatter plot to understand school bus accidents 
# in different hours and light conditions
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(212)
df = accident_factors.loc[accident_factors['school_bus_related'] == 'Yes']
plt.scatter(df.hour_of_crash, df.light_condition_name, color='b')
plt.title("Scatter Plot of Hour vs Light Condition of School Buses...")
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5)
plt.show()
# From the scatter plot we can see that few accidents happened 
# at 3 AM, which seems very odd and require further investigation
# This brings us to the end of our SQL Scavenger Hunt: Wrap-up Analysis
# Thank you!