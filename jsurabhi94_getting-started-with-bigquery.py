% matplotlib inline

import pandas as pd

from bq_helper import BigQueryHelper

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

Chicago_Crime = BigQueryHelper(active_project = "bigquery-public-data", dataset_name = "chicago_crime")
Chicago_Crime.list_tables()
Chicago_Crime.table_schema("crime")
Chicago_Crime.head("crime")
Chicago_Crime.head("crime", selected_columns = ["case_number", "longitude", "latitude"], num_rows = 10)
Query = """SELECT case_number, date, primary_type, location 

            FROM `bigquery-public-data.chicago_crime.crime`"""

Chicago_Crime.estimate_query_size(Query)

different_crimes = Chicago_Crime.query_to_pandas_safe(Query)
different_crimes.head()
different_crimes.set_index("case_number")