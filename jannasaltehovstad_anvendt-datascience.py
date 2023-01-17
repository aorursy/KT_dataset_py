import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import bq_helper
from bq_helper import BigQueryHelper
from google.cloud import bigquery
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_crime")

bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
bq_assistant.list_tables()
bq_assistant.head("crime", num_rows=3)

bq_assistant.table_schema("crime")

query3=  """SELECT year,popular_crimetype from (
select year, max(primary_type) as popular_crimetype
  FROM 
    `bigquery-public-data.chicago_crime.crime`
where primary_type != ""
group by year)
ORDER BY
  year DESC
        """
size = bq_assistant.estimate_query_size(query3)
print("Query size: " + str(size))
all_data = chicago_crime.query_to_pandas_safe(query3)
all_data.head(10)
feature_query = """
                SELECT date,primary_type,location_description
                FROM  `bigquery-public-data.chicago_crime.crime`
                WHERE domestic = FALSE AND year > 2014
                """
size = bq_assistant.estimate_query_size(feature_query)
print("Query size: " + str(size))
dataset = chicago_crime.query_to_pandas_safe(feature_query)
dataset.head(10)
import pandas as pd
import datetime as dt
dateobject = pd.to_datetime(dataset['date']) #Timestamp to date
dataset['date'] = dateobject.dt.date #Returns the date without timezone information
dataset['date'] = dateobject.dt.strftime('%Y-%m-%d') #Format
dataset.head()

focus_words = ['SCHOOL']
dataset['location_description'] = dataset['location_description'].apply(lambda x: str(x).split("/")) #Gjør om til liste av ord
print(dataset.head(10))

#df= dataset['location_description'].filter(lambda x :  x in(focus_words ))

#df = dataset.apply(lambda x: x.str.contains('|'.join(focus_words))).any(1)

#dataset.loc[dataset['location_description'].isin(focus_words), 'school'] = 1
