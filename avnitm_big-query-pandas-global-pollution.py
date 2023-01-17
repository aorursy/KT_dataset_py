import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
world_pop = pd.read_csv('../input/wikipedia-iso-country-codes.csv')
world_pop_codes = world_pop.filter(items=['English short name lower case', 'Alpha-2 code'])
world_pop_codes.columns = ['country_name', 'country']
world_pop_codes.head()
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to pass to 
query = """SELECT country, SUM(value)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY country
            ORDER BY SUM(value) DESC
            LIMIT 5
            
        """
pollution_by_country = open_aq.query_to_pandas_safe(query)
pol_per_country = pd.merge(world_pop_codes, pollution_by_country, on='country', how='inner')
print(pol_per_country)
# query to pass to 
query = """SELECT city, SUM(value)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY city
            ORDER BY SUM(value) DESC
            LIMIT 5 
        """
most_polluted = open_aq.query_to_pandas_safe(query)
print(most_polluted)
# query to find out the max pollution which 
# happen on each day of the week
query = """SELECT MAX(value), 
                  EXTRACT(DAYOFWEEK FROM timestamp)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp)
            ORDER BY MAX(value) DESC
        """
max_pol_per_day = open_aq.query_to_pandas_safe(query)
print(max_pol_per_day)
# query to find out the max pollution which 
# happen on each hour of the day
query = """SELECT MAX(value), 
                  EXTRACT(HOUR FROM timestamp)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY EXTRACT(HOUR FROM timestamp)
            ORDER BY MAX(value) DESC
        """
max_pol_per_hour = open_aq.query_to_pandas_safe(query)
print(max_pol_per_hour)
sns.set(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))

# Plot the orbital period with horizontal boxes
sns.boxplot(x="f1_", y= "f0_", data=max_pol_per_hour,
            whis=np.inf, palette="vlag")

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="Pollution")
ax.set(xlabel="Hour of Day")
sns.despine(trim=True, left=True)