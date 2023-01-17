# Standard Python analytics packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# BigQuery to allow us to use SQL queries to access and export the data

import bq_helper



# kaggle provided imports to allow us to export our work external to kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

plt.rcParams['figure.figsize'] = [12, 6]

        



# Importing the data set we will be analysing via BigQuery

london = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="london_crime")



# This will display the tables contained in the database

london.list_tables()
# First five data entries

london.head('crime_by_lsoa')
# The list of boroughs our data contains, and the amount of boroughs

bo = london.query_to_pandas_safe("""SELECT DISTINCT borough

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

ORDER BY borough;

        """)



bo
min_c = london.query_to_pandas_safe("""

SELECT DISTINCT minor_category, major_category, SUM(value) AS incidents

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

GROUP BY minor_category, major_category

ORDER BY major_category, incidents DESC;

""")

min_c
# Initial BQ query to extract the data into a pandas dataframe...

change = london.query_to_pandas_safe("""

SELECT year, month, SUM(value) AS total_incidents

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

GROUP BY year, month

ORDER BY year, month;

""")



# We will then use the pandas .pct_change method to add a new column containing the percentage change of crime over time

# We will also add a timedate column and set is as the index to allow us to visualise this with Python

change['change_per_month'] = change['total_incidents'].pct_change()

change['date'] = pd.to_datetime(change.year.map(str) + '-' + change.month.map(str), format = '%Y-%m')

change.fillna(0, inplace=True)

change.set_index('date', inplace=True)



# Time series visualisation, displaying both the overall change over the 8 year range, as well as the % change over the range

plt.subplots(1,2,figsize=(14, 6))

plt.subplot(1,2,1)

plt.plot(change.index, change.total_incidents, color='blue')

plt.title('Total Incidents')

plt.subplot(1,2,2)

plt.plot(change.index, change.change_per_month, color='red')

plt.title('% Change in Incidents')

plt.tight_layout()

plt.show()



change
# Import required package from statsmodels

from statsmodels.tsa.seasonal import seasonal_decompose



# Perform decomposition and plot the components of the incidents timeseries

m_decomp_results = seasonal_decompose(change.total_incidents, freq=12, model='additive')

m_decomp_results.plot()

plt.show()
total = london.query_to_pandas_safe("""

SELECT borough, SUM(value) as total_incidents

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

GROUP BY borough

ORDER BY total_incidents DESC;

        """)

total
# Data to populate our new dataframe. This date was extracted from the London Datastore via a simple copy and paste from the website.

# The lists contain population values and thier associated boroughs which we will combine together.

pop = [8800,209000,389600,244300,332100,327900,242500,386500,351600,333000,280100,274300,185300,278000,252300,254300,301000,274200,

231200,159000,175400,328900,303400,208100,342900,304200,197300,314300,202600,304000,276200,321000,242100]



new_b = ['City of London','Barking and Dagenham','Barnet','Bexley','Brent','Bromley','Camden','Croydon','Ealing','Enfield','Greenwich',

'Hackney','Hammersmith and Fulham','Haringey','Harrow','Havering','Hillingdon','Hounslow','Islington','Kensington and Chelsea',

'Kingston upon Thames','Lambeth','Lewisham','Merton','Newham','Redbridge','Richmond upon Thames','Southwark','Sutton','Tower Hamlets',

'Waltham Forest','Wandsworth','Westminster']



# Combining the data in a dictionary which we then create a dataframe with.

data = {'borough': new_b, 'population': pop}

bor_pop = pd.DataFrame(data)



# We then merge our existing dataframe with the new one. The 'inner' join is the default join used by the merge function. 

# We use the borough column as our key.

new = bor_pop.merge(total, on='borough')



# We create a new column in our newly merged dataframe which calculates our arbitrary ratio, which we then sort the data using.

new['ratio'] = new.total_incidents / new.population

new = new.sort_values(by=['ratio'], ascending=False)

new