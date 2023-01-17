import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
aq = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
QUERY = """
    SELECT
        *
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      county_name = "Multnomah"
      AND state_name = "Oregon"
      AND sample_duration = "24 HOUR"
      AND poc = 1
        """
aq.estimate_query_size(QUERY) # GB
df = aq.query_to_pandas(QUERY)
df.columns
df.address.unique()
df.local_site_name.unique()
import datetime
startdate = datetime.date(2015, 1, 1)
enddate = datetime.date(2016, 1, 1)
data_2015 = df[(df.date_local >= startdate) & (df.date_local < enddate)]
data_2015.describe()
data_2015.plot(x='date_local', y='aqi');
data_2015.set_index('date_local').sort_index().head()
data_2015.to_csv('data_2015.csv')
