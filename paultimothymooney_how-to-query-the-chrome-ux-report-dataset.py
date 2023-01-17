import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chromeUXreport = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chrome-ux-report.all")
chromeUXreportUS = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chrome-ux-report.country_us")
chromeUXreportIN = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chrome-ux-report.country_in")
query1 = """SELECT DISTINCT origin
FROM `chrome-ux-report.all.201710`
WHERE origin LIKE '%://example.com';
        """
response1 = chromeUXreport.query_to_pandas_safe(query1)
response1.head(20)
query2 = """SELECT
    bin.start,
    SUM(bin.density) AS density
FROM
    `chrome-ux-report.all.201710`,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    origin = 'http://example.com'
GROUP BY
    bin.start
ORDER BY
    bin.start;
        """
response2 = chromeUXreport.query_to_pandas_safe(query2)
response2.head(20)
import numpy as np 
import pandas as pd 
import os
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

result1 = response2.head(10)
trace1 = go.Bar(
                x = result1.start,
                y = result1.density,
                name = "citations",
                marker = dict(color = 'rgba(0, 0, 255, 0.8)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = result1.start)
data = [trace1]
layout = go.Layout(barmode = "group",title='Density Per Bin', xaxis = dict(title='Start (ms)'), yaxis = dict(title='Density'))
fig = go.Figure(data = data, layout = layout)
iplot(fig)
query3 = """SELECT
    SUM(bin.density) AS density
FROM
    `chrome-ux-report.all.201710`,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    bin.start < 1000 AND
    origin = 'http://example.com';
        """
response3 = chromeUXreport.query_to_pandas_safe(query3)
response3.head(20)
query4 = """#standardSQL
SELECT
    effective_connection_type.name AS ect,
    SUM(bin.density) AS density
FROM
    `chrome-ux-report.all.201710`,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    bin.end <= 1000 AND
    origin = 'http://example.com'
GROUP BY
    ect
ORDER BY
    density DESC;
        """
response4 = chromeUXreport.query_to_pandas_safe(query4)
response4.head(20)
query5 = """WITH
    countries AS (
      SELECT *, 'All' AS country FROM `chrome-ux-report.all.201712`)

SELECT
    country,
    effective_connection_type.name AS ect,
    SUM(bin.density) AS density
FROM
    countries,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    bin.end <= 1000 AND
    origin = 'http://example.com'
GROUP BY
    country,
    ect
ORDER BY
    density DESC;
        """
response5 = chromeUXreport.query_to_pandas_safe(query5, max_gb_scanned=10)
response5.head(20)
query6 = """WITH
    countries AS (
      SELECT *, 'USA' AS country FROM `chrome-ux-report.country_us.201712`)

SELECT
    country,
    effective_connection_type.name AS ect,
    SUM(bin.density) AS density
FROM
    countries,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    bin.end <= 1000 AND
    origin = 'http://example.com'
GROUP BY
    country,
    ect
ORDER BY
    density DESC;
        """
response6 = chromeUXreportUS.query_to_pandas_safe(query6, max_gb_scanned=10)
response6.head(20)
query7 = """WITH
    countries AS (
      SELECT *, 'India' AS country FROM `chrome-ux-report.country_in.201712`)

SELECT
    country,
    effective_connection_type.name AS ect,
    SUM(bin.density) AS density
FROM
    countries,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    bin.end <= 1000 AND
    origin = 'http://example.com'
GROUP BY
    country,
    ect
ORDER BY
    density DESC;
        """
response7 = chromeUXreportIN.query_to_pandas_safe(query7, max_gb_scanned=10)
response7.head(20)