import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset
ds = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                dataset_name = "openaq")
ds.list_tables()
ds.table_schema('global_air_quality')
ds.head('global_air_quality')
ds.head('global_air_quality', selected_columns='city',num_rows=10)
# this query looks in the full table in the global_air_quality
# dataset, then gets the value column from every row where 
# the pollutant column has "pm25" in it.
query = """SELECT value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant = "pm25" """

# check how big this query will be (in GB)
ds.estimate_query_size(query)
# Query
# only run this query if it's less than 100 MB   (dummed to fail! :) )
ds.query_to_pandas_safe(query, max_gb_scanned=0.0002)
# w/o restrictions
adf = ds.query_to_pandas(query)
#get mean
adf.mean()
adf.to_csv('global_air_quality_values.csv')