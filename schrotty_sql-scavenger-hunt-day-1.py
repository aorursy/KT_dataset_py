#Load BigQuery Helper: https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper
# And Pandas:
import pandas as pd
# Create Big Query Helper Object
bq_day1 = BigQueryHelper("bigquery-public-data", "openaq")
# Lets view the 10 first Rows of the Dataset
bq_day1.head("global_air_quality", num_rows=10)
QUERY = """
        SELECT gaq.country, count(*) as country_count
        FROM `bigquery-public-data.openaq.global_air_quality` AS gaq
        WHERE gaq.unit != 'ppm'
        GROUP BY gaq.country
        ORDER BY gaq.country desc   
        """
# Estimate Query size:
bq_day1.estimate_query_size(QUERY)
# Excecute the SQL (safely, not more than 1GB) and put the Result set into a pandas Dataframe (df)
df = bq_day1.query_to_pandas_safe(QUERY)
#lets show the Datframe
df.head(100)
QUERY2 = """
        SELECT gaq.pollutant
        FROM `bigquery-public-data.openaq.global_air_quality` AS gaq
        WHERE gaq.value = 0
        GROUP BY gaq.pollutant
        ORDER BY gaq.pollutant desc   
        """
# Estimate Query #2 size:
bq_day1.estimate_query_size(QUERY2)
# Excecute the Query #2 (safely, not more than 1GB) and put the Result set into a pandas Dataframe (df2)
df2 = bq_day1.query_to_pandas_safe(QUERY2)
#lets show the Datframe
df2.head(100)