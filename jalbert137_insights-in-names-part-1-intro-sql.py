# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# The above is the default set of imports and boilerplate.
# It isn't very helpful, because the data isn't actually in the ../input directory.
# The data is stored with BigQuery, a Google-related infrastructure as a service.
# https://cloud.google.com/bigquery/
# Thus, we need to load the library to help us access it.
# We will follow recipes from...
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package/code
import bq_helper # should let us access the data

# some inspiration is taken from...
#https://www.kaggle.com/salil007/a-very-extensive-exploratory-analysis-usa-names/code

usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
usa_names.list_tables()
usa_names.head("usa_1910_2013", num_rows=10)
# Convert into a pandas dataframe, using the bq_helper, querying to get all fields.
# We need to use SQL-style queries.
# First, see how big it'll be (note the output will be in GB)
QUERY1 = "SELECT * FROM `bigquery-public-data.usa_names.usa_1910_current`"
print("Current database will use", str(usa_names.estimate_query_size(QUERY1)), "GB")
QUERY2 = "SELECT * FROM `bigquery-public-data.usa_names.usa_1910_2013`"
print("2013 database will use", str(usa_names.estimate_query_size(QUERY2)), "GB")
QUERY1 = "SELECT COUNT(DISTINCT year) FROM `bigquery-public-data.usa_names.usa_1910_current`"
yearcount1 = usa_names.query_to_pandas(QUERY1)
print("The 'current' database...")
yearcount1.head()
QUERY2 = "SELECT COUNT(DISTINCT year) FROM `bigquery-public-data.usa_names.usa_1910_2013`"
yearcount2 = usa_names.query_to_pandas(QUERY2)
print("The '2013' database...")
yearcount2.head()
# Lots of "*aiden" names, let's look at some!
# first the basic selection
QUERYa = "SELECT name, gender, SUM(number) AS total FROM `bigquery-public-data.usa_names.usa_1910_2013` "
# filter the dates, states, and gender
QUERYb = "WHERE name LIKE '%aiden' "
QUERYc = "GROUP BY name, gender "
QUERYd = "ORDER BY total DESC"
QUERY = QUERYa + QUERYb + QUERYc + QUERYd
aidentable = usa_names.query_to_pandas(QUERY)
aidentable.head(10)
# Get the most popular girls names in Alaska and Hawaii in the 10 years after they were
# admitted to the USA (1959-1968)
# first the basic selection
QUERYa = "SELECT name, SUM(number) AS total FROM `bigquery-public-data.usa_names.usa_1910_2013` "
# filter the dates, states, and gender
QUERYb = "WHERE (state = 'AK' OR state = 'HI') AND gender = 'F' AND year BETWEEN 1959 AND 1968 "
# grouping
QUERYc = "GROUP BY name "
QUERYd = "ORDER BY total DESC"
QUERY = QUERYa + QUERYb + QUERYc + QUERYd
newstatestable = usa_names.query_to_pandas(QUERY)
newstatestable.head(10)
# Number of distinct boys names vs number of distinct girls names each year
# Of course, we keep in mind that we truncate names below 5, so let's not draw
# any false conclusions.  Instead, let's learn SQL.
# part of me would like to do more here, but let's move on
# first the basic selection
QUERYa = "SELECT year, COUNT(DISTINCT name) AS uniques FROM `bigquery-public-data.usa_names.usa_1910_2013` "
# filter the dates, states, and gender
QUERYb = "WHERE gender = 'F' AND year BETWEEN 1960 AND 1979 "
# grouping
QUERYc = "GROUP BY year "
QUERYd = "ORDER BY year ASC"
QUERY = QUERYa + QUERYb + QUERYc + QUERYd
uniquetable = usa_names.query_to_pandas(QUERY)
uniquetable.head(20)
# Same for boys
QUERYa = "SELECT year, COUNT(DISTINCT name) AS uniques FROM `bigquery-public-data.usa_names.usa_1910_2013` "
# filter the dates, states, and gender
QUERYb = "WHERE gender = 'M' AND year BETWEEN 1960 AND 1979 "
# grouping
QUERYc = "GROUP BY year "
QUERYd = "ORDER BY year ASC"
QUERY = QUERYa + QUERYb + QUERYc + QUERYd
uniquetable = usa_names.query_to_pandas(QUERY)
uniquetable.head(20)