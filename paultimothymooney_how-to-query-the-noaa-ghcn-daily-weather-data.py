import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
noaa = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="ghcn_d")
bq_assistant = BigQueryHelper("bigquery-public-data", "ghcn_d")
bq_assistant.list_tables()
bq_assistant.head("ghcnd_2018", num_rows=3)
bq_assistant.table_schema("ghcnd_2018")
query1 = """SELECT
  id,
  name,
  state,
  latitude,
  longitude
FROM
  `bigquery-public-data.ghcn_d.ghcnd_stations`
WHERE
  latitude > 41.7
  AND latitude < 42
  AND longitude > -87.7
  AND longitude < -87.5;
        """
response1 = noaa.query_to_pandas_safe(query1)
response1.head(10)
query2 = """SELECT
  wx.date,
  wx.value/10.0 AS prcp
FROM
  `bigquery-public-data.ghcn_d.ghcnd_2015` AS wx
WHERE
  id = 'USW00094846'
  AND qflag IS NULL
  AND element = 'PRCP'
ORDER BY wx.date;
        """
response2 = noaa.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(10)
query3 = """SELECT
  date,
  MAX(prcp) AS prcp,
  MAX(tmin) AS tmin,
  MAX(tmax) AS tmax
FROM (
  SELECT
    wx.date AS date,
    IF (wx.element = 'PRCP', wx.value/10, NULL) AS prcp,
    IF (wx.element = 'TMIN', wx.value/10, NULL) AS tmin,
    IF (wx.element = 'TMAX', wx.value/10, NULL) AS tmax
  FROM
    `bigquery-public-data.ghcn_d.ghcnd_2018` AS wx
  WHERE
    id = 'USW00094846'
    AND DATE_DIFF(CURRENT_DATE(), wx.date, DAY) < 15
)
GROUP BY
  date
ORDER BY
  date ASC;
        """
response3 = noaa.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)
