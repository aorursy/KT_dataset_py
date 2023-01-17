import numpy as np

import pandas as pd

import bq_helper

import matplotlib.pyplot as plt

import seaborn as sns
london = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="london_crime")
# Query to select Hillingdons's crime stats by year and month

hillingdon_crime_query = """

SELECT year, month, sum(value) AS `total_crime`

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

WHERE borough = 'Hillingdon'

GROUP BY year, month;

        """



# Perform and store the query results 

hillingdon_crime = london.query_to_pandas_safe(hillingdon_crime_query)

hillingdon_crime.head()
hillingdon_crime.describe().total_crime
hillingdon_crime.index
hillingdon_crime['date'] = pd.to_datetime(hillingdon_crime.year.map(str) + '-' + hillingdon_crime.month.map(str), format = '%Y-%m')

hillingdon_crime.set_index('date', inplace=True)
hillingdon_crime.total_crime.plot(figsize=(15, 6))

plt.title('Total crime in Camden')

plt.ylabel('Total crime per month')

plt.xlabel('')

plt.show()
period_1417 = hillingdon_crime.loc['2014':'2017',]
period_1417.total_crime.plot(figsize=(15, 6))

plt.title('Total crime in Hillingdon between 2014-2017')

plt.ylabel('Total crime per month')

plt.show()
hillingdon_major_query = """

SELECT year, month, major_category, sum(value) AS `total_crime`

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

WHERE borough = 'Hillingdon'

GROUP BY year, month, major_category

ORDER BY year, month;

        """

# Perform and store the query results

hillingdon_major = london.query_to_pandas_safe(hillingdon_major_query)

hillingdon_major.head()
hillingdon_major['date'] = pd.to_datetime(hillingdon_major.year.map(str) + '-' + hillingdon_major.month.map(str), format = '%Y-%m')

hillingdon_major.drop(columns = ['year', 'month'], inplace = True)

hillingdon_major.head()
hillingdon_major_pivot = hillingdon_major.pivot(index = 'date', columns = 'major_category', values = 'total_crime')
hillingdon_major_pivot.plot(subplots = True, figsize=(15, 15))

plt.show()