# Importing Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tabulate import tabulate

import bq_helper

import os
# Importing Data



#https://www.kaggle.com/salil007/a-very-extensive-exploratory-analysis-usa-names

usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

query = """SELECT year, gender, name, sum(number) as count FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""

data = usa_names.query_to_pandas_safe(query)

data.to_csv("usa_names_data.csv")
# What are the most common names?



data['name'].value_counts().head()
# What are the most common female names?



data[data['gender']=='F']['name'].value_counts().head()
# What are the most common male names?



data[data['gender']=='M']['name'].value_counts().head()
# Are there more female or male names?



# data.groupby([data.gender]).count().plot(kind='bar')

data.groupby('gender')['gender'].count().plot.bar()
# No. of applicants per year



applicants_per_year=data.groupby('year')['year'].count().plot.line()

applicants_per_year.set_ylabel("Number of applicants")