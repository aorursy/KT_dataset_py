from google.cloud import bigquery
from bq_helper import BigQueryHelper
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
bq_assistant = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")
client = bigquery.Client()

%%time
#list tables

bq_assistant.list_tables()


%%time
bq_assistant.table_schema("vehicle_2016")
QUERY = """SELECT
  vehicle_make_name ,vehicle_model, registration_state_name,
  COUNT(consecutive_number) AS accidents,
  SUM(fatalities_in_vehicle) AS fatalities,
  SUM(fatalities_in_vehicle) / COUNT(consecutive_number) AS fatalities_per_accident
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_{0}`
GROUP BY
  vehicle_make_name,vehicle_model,registration_state_name
ORDER BY
  fatalities_per_accident DESC
  """

bq_assistant.estimate_query_size(QUERY.format(2015))
%%time
bq_assistant.head("vehicle_2016", num_rows=10)
%%time

bq_assistant.estimate_query_size(QUERY.format(2015))
%%time
df1 = bq_assistant.query_to_pandas_safe(QUERY.format(2015))
df1.head()
%%time
df2 = bq_assistant.query_to_pandas_safe(QUERY.format(2016))
df2.head()