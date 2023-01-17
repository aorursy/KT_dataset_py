# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.





# plotting libraries:

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

%matplotlib inline





# Google Big Query library:

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

import bq_helper

from bq_helper import BigQueryHelper
# allowing for any single variable to print out without using the print statement:

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



sns.set(color_codes=True) #overide maplot libs ugly colours.

mpl.rcParams['figure.figsize'] = [13, 8] #default figure size
# creating an instance of the database that you can send SQL queries to later and get data back

chicago_taxi = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="chicago_taxi_trips")
# only one table in this "database"

chicago_taxi.list_tables()
# seeing some data from this table

chicago_taxi.head("taxi_trips", num_rows=3)
# seeing the table schema of this to see more details of our table

chicago_taxi.table_schema("taxi_trips")
# let's see how many rows we have:

query = """SELECT

  COUNT(*) as Number_of_Rows

FROM

  `bigquery-public-data.chicago_taxi_trips.taxi_trips`

        """

response = chicago_taxi.query_to_pandas_safe(query, max_gb_scanned=10)

"Number of rows in this dataset:"

print(response)
query1 = """SELECT taxi_id,

    SUM(fare) as yearly_pay,

    EXTRACT(YEAR FROM trip_start_timestamp) AS year

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY year, taxi_id;

"""



response1 = chicago_taxi.query_to_pandas_safe(query1, max_gb_scanned=16)

response1.head()
# turning the yearly_pay from cents to dollars and cents

response1.yearly_pay = round(response1.yearly_pay / 100, 2)

response1.head()
# average yearly pay:

"Average Yearly Pay with outliers:"

response1.groupby('year').yearly_pay.mean()
# Searching for outliers that might skew our data:

"Number of Taxi Drivers Per Year in our Dataset:"

response1.year.value_counts(dropna=False).sort_index()
response1 = response1[response1.year.between(2014, 2016)]
'Number of Taxi Drivers Working Per Year'

response1.year.value_counts(dropna=False).sort_index()
# searching for more outliers - let's look at a boxplot

sns.boxplot(data = response1,x="year",y="yearly_pay")
# looking at the numbers falling within 0 and 175,000 dollars:

"Average Yearly Pay without Outliers:"

response1[response1.yearly_pay.between(0, 175000)].groupby('year').yearly_pay.mean()

response1[response1.yearly_pay.between(0, 175000)].groupby('year').yearly_pay.mean().plot('bar')
sns.boxplot(data = response1,x="year",y="yearly_pay", showfliers=False).set_title('Average Yearly Pay - Taxi Drivers in Chicago')
# How many rides were there in 2015?

query = """SELECT COUNT(*) as number_of_rides_2015

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2015;

"""



response = chicago_taxi.query_to_pandas_safe(query, max_gb_scanned=22)

print(response)
# How many rides were there in 2016?

query = """SELECT COUNT(*) as number_of_rides_2016

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2016;

"""



response = chicago_taxi.query_to_pandas_safe(query, max_gb_scanned=22)

print(response)
# What does the data look like per month?

query = """SELECT COUNT(*) as number_of_rides,

    DATE_TRUNC(DATE(trip_start_timestamp), WEEK) as date

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY date

ORDER BY date ASC;

"""



response = chicago_taxi.query_to_pandas_safe(query, max_gb_scanned=22)

print(response)
ax = sns.lineplot(x="date", y="number_of_rides", data=response)

ax.set(title="Number of Chicago Taxi Rides per Week 2013-2017")