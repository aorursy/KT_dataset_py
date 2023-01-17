from mpl_toolkits.basemap import Basemap

import folium

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import pycountry

import datetime

import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams

rcParams['figure.figsize'] = 10, 10

from google.cloud import bigquery

# Create a "Client" object

client = bigquery.Client()

# Construct a reference to dataset

dataset_ref = client.dataset("noaa_gsod", project="bigquery-public-data")

# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)
# Construct a reference to the "full" table

table_ref = dataset_ref.table("gsod2019")



# API request - fetch the table

table = client.get_table(table_ref)
# Print information on all the columns

table.schema
# Preview the first five lines of the "full" table

client.list_rows(table, max_results=5).to_dataframe()