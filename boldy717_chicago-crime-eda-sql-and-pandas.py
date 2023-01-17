import bq_helper
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.pyplot import figure
from bq_helper import BigQueryHelper
import pandas as pd

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_crime")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
#List all availablt tables
bq_assistant.list_tables()
# First couple rows of the crime table
bq_assistant.head("crime", num_rows=3)
# Schema and dtypes of the variables in the crime table
bq_assistant.table_schema("crime")
# Query construction to look at macro trends (total incidents reported, arrests, and percent of arrests) over the years
query_all = """SELECT year,
                COUNT(*) AS Incidents,
                COUNTIF(arrest = TRUE) AS Arrests,
                COUNTIF(arrest = TRUE) / COUNT(*) AS Pct_Arrested
            FROM `bigquery-public-data.chicago_crime.crime`
            GROUP BY year
            ORDER BY year"""
# Run query and store in pandas df
response_all = chicago_crime.query_to_pandas_safe(query_all)

# Set plot size and text size
fig, ax1 = plt.subplots(figsize=(15,8))
plt.rcParams['font.size'] = 15

#Create plot with bars for incidents and arrests
ax1.set_xlabel('Year')
ax1.set_ylabel('Count')
ax1.bar(x = response_all.year, height = response_all.Incidents, label = "Incidents")
ax1.bar(x = response_all.year, height = response_all.Arrests, label = "Arrests")
plt.yticks(np.arange(0, 650000, 100000))
# Set correct tick marks on x-axis
plt.xticks(np.arange(min(response_all.year), max(response_all.year)+1, step=1), rotation = 'vertical')
ax1.legend(loc=2)
# Create second axis with the same x-axis for percent arrested
ax2 = ax1.twinx() 
# Create line for percent arrested over time
color = 'tab:red'
ax2.set_ylabel('% Arrested')
ax2.plot(response_all.year, response_all.Pct_Arrested*100, color = color, label = "% Arrested", linewidth = 3)
ax2.legend(loc=0)

plt.show()
# Construct a query to get a better understanding of most common types of incident and arrests and their percentages of arrests. 
query_type_count = """SELECT primary_type, 
                            COUNT(*) AS Incidents, 
                            COUNTIF(arrest = TRUE) AS Arrests, 
                            (COUNTIF(arrest = TRUE) / COUNT(*)) * 100 AS Percent_Arrests 
                    FROM `bigquery-public-data.chicago_crime.crime` 
                    GROUP BY primary_type 
                    ORDER BY Arrests DESC"""
chicago_crime.query_to_pandas_safe(query_type_count)
# Initial query to get incidents by type over time
query_incidents = """SELECT year,
                    primary_type,
                    COUNT(*) AS Incidents
                FROM `bigquery-public-data.chicago_crime.crime`
                GROUP BY year, primary_type"""
df_incidents = chicago_crime.query_to_pandas_safe(query_incidents)
# Initial query to get arrests by type over time
query_arrest = """SELECT year,
                    primary_type,
                    COUNT(*) AS Arrests
                FROM `bigquery-public-data.chicago_crime.crime`
                WHERE arrest = TRUE
                GROUP BY year, primary_type"""
df_arrests = chicago_crime.query_to_pandas_safe(query_arrest)

# Using df in previous cell, create lists of most common types of incidents and arrests, this will be used to filter the query results to remove less significant types
# PS if anyone has ideas of how to integrate this filter step into the query, perhaps using RANK OVER PARTITION, please let me know. 
top_incidents = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'OTHER OFFENSE', 'ASSAULT', 'BURGLARY', 'MOTOR VEHICLE THEFT', 'DECEPTIVE PRACTICE']
top_arrests = ['NARCOTICS', 'BATTERY', 'BATTERY', 'CRIMINAL TRESSPASS', 'ASSAULT','OTHER OFFENSE', 'PROSTITUTION', 'WEAPONS VIOLATION', 'CRIMINAL DAMAGE']
# Use the lists to filter the query responses
df_incidents = df_incidents.loc[df_incidents['primary_type'].isin(top_incidents)]
df_arrests = df_arrests.loc[df_arrests['primary_type'].isin(top_arrests)]

#Set plot figure size and text size
plt.rcParams['font.size'] = 22
fig = plt.figure(figsize = (25, 30))

# First plot, count of incidents by type
ax1 = fig.add_subplot(211)
ax1.set_title("Count of Incidents by Type")
# Set correct tick marks on x-axis
plt.xticks(np.arange(min(df_incidents.year), max(df_incidents.year)+1, step=1))
# Use for loop to apply name of incident as label to each type's line
for name, group in df_incidents.groupby('primary_type'):
    group.plot(x='year', y='Incidents', ax=ax1, label=name)
ax1.legend(loc=0, fontsize='small')
# Second plot, count of arrests by type
ax2 = fig.add_subplot(212)
ax2.set_title("Count of Arrests by Type")
# Set correct tick marks on x-axis
plt.xticks(np.arange(min(df_arrests.year), max(df_arrests.year)+1, step=1))
# Use for loop to apply name of incident as label to each type's line
for name, group in df_arrests.groupby('primary_type'):
    group.plot(x='year', y='Arrests', ax=ax2, label=name)
ax2.legend(loc=0, fontsize='small')
# Construct query to calculate the % change for each year by the type of arrest
query_pct_change_arrest = """SELECT
  primary_type,
  FORMAT('%3.2f', (COUNTIF(year = 2002) - COUNTIF(year = 2001)) / CASE WHEN COUNTIF(year = 2001) = 0 THEN NULL ELSE COUNTIF(year = 2001) END) AS d2002,
  FORMAT('%3.2f', (COUNTIF(year = 2003) - COUNTIF(year = 2002)) / CASE WHEN COUNTIF(year = 2002) = 0 THEN NULL ELSE COUNTIF(year = 2002) END) AS d2003,
  FORMAT('%3.2f', (COUNTIF(year = 2004) - COUNTIF(year = 2003)) / CASE WHEN COUNTIF(year = 2004) = 0 THEN NULL ELSE COUNTIF(year = 2004) END) AS d2004,
  FORMAT('%3.2f', (COUNTIF(year = 2005) - COUNTIF(year = 2004)) / CASE WHEN COUNTIF(year = 2005) = 0 THEN NULL ELSE COUNTIF(year = 2005) END) AS d2005,
  FORMAT('%3.2f', (COUNTIF(year = 2006) - COUNTIF(year = 2005)) / CASE WHEN COUNTIF(year = 2006) = 0 THEN NULL ELSE COUNTIF(year = 2006) END) AS d2006,
  FORMAT('%3.2f', (COUNTIF(year = 2007) - COUNTIF(year = 2006)) / CASE WHEN COUNTIF(year = 2007) = 0 THEN NULL ELSE COUNTIF(year = 2007) END) AS d2007,
  FORMAT('%3.2f', (COUNTIF(year = 2008) - COUNTIF(year = 2007)) / CASE WHEN COUNTIF(year = 2008) = 0 THEN NULL ELSE COUNTIF(year = 2008) END) AS d2008,
  FORMAT('%3.2f', (COUNTIF(year = 2009) - COUNTIF(year = 2008)) / CASE WHEN COUNTIF(year = 2009) = 0 THEN NULL ELSE COUNTIF(year = 2009) END) AS d2009,
  FORMAT('%3.2f', (COUNTIF(year = 2010) - COUNTIF(year = 2009)) / CASE WHEN COUNTIF(year = 2010) = 0 THEN NULL ELSE COUNTIF(year = 2010) END) AS d2010,
  FORMAT('%3.2f', (COUNTIF(year = 2011) - COUNTIF(year = 2010)) / CASE WHEN COUNTIF(year = 2011) = 0 THEN NULL ELSE COUNTIF(year = 2011) END) AS d2011,
  FORMAT('%3.2f', (COUNTIF(year = 2012) - COUNTIF(year = 2011)) / CASE WHEN COUNTIF(year = 2012) = 0 THEN NULL ELSE COUNTIF(year = 2012) END) AS d2012,
  FORMAT('%3.2f', (COUNTIF(year = 2013) - COUNTIF(year = 2012)) / CASE WHEN COUNTIF(year = 2013) = 0 THEN NULL ELSE COUNTIF(year = 2013) END) AS d2013,
  FORMAT('%3.2f', (COUNTIF(year = 2014) - COUNTIF(year = 2013)) / CASE WHEN COUNTIF(year = 2014) = 0 THEN NULL ELSE COUNTIF(year = 2014) END) AS d2014,
  FORMAT('%3.2f', (COUNTIF(year = 2015) - COUNTIF(year = 2014)) / CASE WHEN COUNTIF(year = 2015) = 0 THEN NULL ELSE COUNTIF(year = 2015) END) AS d2015,
  FORMAT('%3.2f', (COUNTIF(year = 2016) - COUNTIF(year = 2015)) / CASE WHEN COUNTIF(year = 2016) = 0 THEN NULL ELSE COUNTIF(year = 2016) END) AS d2016,
  FORMAT('%3.2f', (COUNTIF(year = 2017) - COUNTIF(year = 2016)) / CASE WHEN COUNTIF(year = 2017) = 0 THEN NULL ELSE COUNTIF(year = 2017) END) AS d2017
FROM
  `bigquery-public-data.chicago_crime.crime`
WHERE
  arrest = TRUE
GROUP BY
  primary_type
ORDER BY COUNTIF(year = 2018) DESC"""

response_pct_change_arrest = chicago_crime.query_to_pandas_safe(query_pct_change_arrest)
response_pct_change_arrest
query_narcotics_pct = """SELECT primary_type,
                            COUNTIF(year = 2015) AS Incidents_2015,
                            COUNTIF(year = 2015 AND arrest = TRUE) AS Arrests_2015,
                            ( COUNTIF(year = 2015 AND arrest = TRUE)/ CASE WHEN COUNTIF(year = 2015) = 0 THEN NULL ELSE COUNTIF(year = 2015) END ) * 100 AS Pct_2015,
                            COUNTIF(year = 2016) AS Incidents_2016,
                            COUNTIF(year = 2016 AND arrest = TRUE) AS Arrests_2016,
                            ( COUNTIF(year = 2016 AND arrest = TRUE)/ CASE WHEN COUNTIF(year = 2016) = 0 THEN NULL ELSE COUNTIF(year = 2016) END ) * 100 AS Pct_2016,
                            ( ( ( COUNTIF(year = 2016 AND arrest = TRUE)/ CASE WHEN COUNTIF(year = 2016) = 0 THEN NULL ELSE COUNTIF(year = 2016) END ) -
                                ( COUNTIF(year = 2015 AND arrest = TRUE)/ CASE WHEN COUNTIF(year = 2015) = 0 THEN NULL ELSE COUNTIF(year = 2015) END ) ) /
                                    ( COUNTIF(year = 2015 AND arrest = TRUE)/ CASE WHEN COUNTIF(year = 2015) = 0 THEN NULL ELSE COUNTIF(year = 2015) END ) ) * 100 AS Pct_Delta
                            FROM `bigquery-public-data.chicago_crime.crime`
                            GROUP BY primary_type
                            ORDER BY Incidents_2016 DESC"""

pct_delt = chicago_crime.query_to_pandas_safe(query_narcotics_pct)
# Filter resulting df for only top 15 rows for the bar plot
pct_delt = pct_delt.head(15)

#Set plot figure size and text size
plt.rcParams['font.size'] = 11
fig = plt.figure(figsize = (17, 25))
# Create horizontal barchart for % deltas
fig, ax = plt.subplots()
ax.barh(pct_delt.primary_type, pct_delt.Pct_Delta)
# Lables and show plot + full df
ax.set_xlabel('Percent Delta')
ax.set_title('Pct Delta for Pct of Arrests of Top 10 types by Count of Incidents in 2016')
plt.show()
chicago_crime.query_to_pandas_safe(query_narcotics_pct)
# Query to get count of incidents of type battery, narcotics, and theft by hour of day
query_hour_type = """SELECT EXTRACT(HOUR FROM date)  AS Hour,
                        COUNTIF(year = 2018 AND primary_type = 'BATTERY') AS Count_Battery,
                        COUNTIF(year = 2018 AND primary_type = 'NARCOTICS') AS Count_Narcotics,
                        COUNTIF(year = 2018 AND primary_type = 'THEFT') AS Count_Theft
                FROM `bigquery-public-data.chicago_crime.crime`
                GROUP BY Hour
                ORDER BY Hour"""
hour_type = chicago_crime.query_to_pandas_safe(query_hour_type)

# Set plot figure size
fig = plt.figure(figsize = (20, 7))

# Create subplots for each type of incident

ax = fig.add_subplot(131)
ax.bar(hour_type.Hour, hour_type.Count_Battery)
ax.set_title('Battery Incidents By Hour of Day')
plt.xlabel('Hour')

ax2 = fig.add_subplot(132)
ax2.bar(hour_type.Hour, hour_type.Count_Narcotics)
ax2.set_title('Narcotics Incidents By Hour of Day')
plt.xlabel('Hour')

ax3 = fig.add_subplot(133)
ax3.bar(hour_type.Hour, hour_type.Count_Theft)
ax3.set_title('Theft Incidents By Hour of Day')
plt.xlabel('Hour')

plt.show()