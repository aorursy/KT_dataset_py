import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
# sample query from:
# https://cloud.google.com/bigquery/public-data/openaq#which_10_locations_have_had_the_worst_air_quality_this_month_as_measured_by_high_pm10
QUERY = """
        SELECT location, city, country, value, timestamp
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = "pm10" AND timestamp > "2017-04-01"
        ORDER BY value DESC
        LIMIT 1000
        """
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(QUERY)
df.head(3)
client = bigquery.Client()
query_job = client.query(QUERY)
rows = list(query_job.result(timeout=30))
for row in rows[:3]:
    print(row)
type(rows[0])
list(rows[0].keys())
list(rows[0].values())
df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df.head(3)
df.info()
df['value'].plot()
QUERY_1 = """
        SELECT unit, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != "ppm" 
        LIMIT 1000
        """
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(QUERY_1)
df.head(10)
client = bigquery.Client()
query_job = client.query(QUERY_1)
rows = list(query_job.result(timeout=30))
for row in rows[:3]:
    print(row)

print(list(rows[0].keys()))
print(list(rows[0].values()))

df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df.head(10)