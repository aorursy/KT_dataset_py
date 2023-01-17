import numpy as np
import pandas as pd
import bq_helper
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
%matplotlib inline
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
# What ten cities have the most measurements taken there?
query1 = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
us_cities = open_aq.query_to_pandas_safe(query1)
us_cities.city.value_counts().head(10)
# Which U.S. city has the highest pm2.5 value in average?
query2 = """SELECT city, pollutant, AVG(value) AS average_value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US' AND pollutant = 'pm25'
            GROUP BY city,pollutant
            ORDER BY average_value DESC
            LIMIT 10
        """
high_pm25_city = open_aq.query_to_pandas_safe(query2)
high_pm25_city
plt.subplots(figsize=(15,8))
sns.barplot(x='city',y='average_value',data=high_pm25_city,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('PM2.5 values in µg/m³', fontsize=20)
plt.xticks(rotation=360)
plt.xlabel('city', fontsize=20)
plt.title('Average value of PM2.5 in different US cities', fontsize=24)
plt.show()
hacker_news = bq_helper.BigQueryHelper("bigquery-public-data", "hacker_news")
hacker_news.head("comments")
popular_comments = hacker_news.query_to_pandas_safe("""
    SELECT parent, COUNT(id) AS num_replies
    FROM `bigquery-public-data.hacker_news.comments`
    GROUP BY parent
    HAVING COUNT(id) > 500
    ORDER BY 2 DESC
""")
popular_comments.head()
stories_by_type = hacker_news.query_to_pandas_safe("""
    SELECT type, COUNT(id) AS num_of_type
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
""")
stories_by_type
accidents = bq_helper.BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")
print(accidents.list_tables())
accidents_by_day = accidents.query_to_pandas_safe("""
    SELECT COUNT(consecutive_number) AS crash_count,
           EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS week_day
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY week_day
    ORDER BY crash_count DESC
""")
accidents_by_day
plt.plot(accidents_by_day.crash_count)
plt.title('Accidents by day 2016')
accidents_by_hour = accidents.query_to_pandas_safe("""
    SELECT COUNT(consecutive_number) AS crash_count,
           EXTRACT(HOUR FROM timestamp_of_crash) AS hour
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY hour
    ORDER BY crash_count DESC
""")
accidents_by_hour
hit_and_run = accidents.query_to_pandas_safe("""
    SELECT registration_state_name AS state,
           COUNT(consecutive_number) AS number
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
    WHERE hit_and_run = 'Yes'
    GROUP BY state
    ORDER BY number DESC
""")
hit_and_run.head(7)