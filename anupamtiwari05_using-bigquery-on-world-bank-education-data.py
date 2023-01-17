import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Importing python library to work with BigQuery
import bq_helper
# Setting up a database instance
worldBank_educationdata = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "world_bank_intl_education")
# Exploring tables in our dataset
worldBank_educationdata.list_tables()
# Below is helpful to check table schema to know more about the table and columns it has along-with data types and other
# restrictions imposed
worldBank_educationdata.table_schema("series_summary")
# Use .head("table_name",selected_columns="coulmn_name",num_rows="No._of_recordstoshow") with you BigQueryHelper to show 
# top 5 records if you skip second parameter however you can provide the second parameter to check as many records as 
# you want. Also you have the facility to show just one or more columns worldBank_educationdata.head("series_summary",3)

worldBank_educationdata.head("series_summary",selected_columns=("series_code","topic"),num_rows = 3)
# Now running a select query to get unique topic where topic length is less than 20
query = """select distinct topic 
        from `bigquery-public-data.world_bank_intl_education.series_summary`
        where length(topic)<20"""
worldBank_educationdata.query_to_pandas_safe(query)