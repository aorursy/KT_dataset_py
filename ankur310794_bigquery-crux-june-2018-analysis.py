import bq_helper
from bq_helper import BigQueryHelper

import numpy as np 
import pandas as pd 
import os
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import seaborn as sns
init_notebook_mode(connected=True)
color = sns.color_palette()

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('figure', figsize=(10, 8))
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chromeUXreport = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chrome-ux-report.all")
chromeUXreportUS = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chrome-ux-report.country_us")
chromeUXreportIN = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chrome-ux-report.country_in")
query1 = """SELECT
    bin.start,
    SUM(bin.density) AS density
FROM
    `chrome-ux-report.all.201806`,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    origin = 'https://www.kaggle.com'
GROUP BY
    bin.start
ORDER BY
    bin.start;
        """

print(chromeUXreport.estimate_query_size(query1))
response1 = chromeUXreport.query_to_pandas_safe(query1, max_gb_scanned= 5)
response1.head(20)
result1 = response1.head(10)
trace1 = go.Bar(
                x = result1.start,
                y = result1.density,
                name = "citations",
                marker = dict(color = 'rgba(0, 0, 255, 0.8)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = result1.start)
data = [trace1]
layout = go.Layout(barmode = "group",title='First Contentful Paint Density Per Bin', xaxis = dict(title='Start (ms)'), yaxis = dict(title='Density'))
fig = go.Figure(data = data, layout = layout)
iplot(fig)
query2 = """SELECT
    SUM(bin.density) AS density
FROM
    `chrome-ux-report.all.201806`,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    bin.start < 5000 AND
    origin = 'https://www.kaggle.com';
        """
print(chromeUXreport.estimate_query_size(query2))
response2 = chromeUXreport.query_to_pandas_safe(query2,max_gb_scanned=5)
response2.head(20)
query3 = """
#standardSQL
SELECT
    effective_connection_type.name AS ect,
    SUM(bin.density) AS density
FROM
    `chrome-ux-report.all.201806`,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    bin.end <= 5000 AND
    origin = 'https://www.kaggle.com'
GROUP BY
    ect
ORDER BY
    density DESC;
        """
print(chromeUXreport.estimate_query_size(query3))
response3 = chromeUXreport.query_to_pandas_safe(query3,max_gb_scanned=5)
response3.head(20)
result3 = response3
sns.factorplot(x='ect', y='density', data=result3, kind='bar', size=4, aspect=2.0)
query4 = """
#standardSQL
WITH
    countries AS (
      SELECT *, 'All' AS country FROM `chrome-ux-report.all.201806`
    UNION ALL
      SELECT *, 'India' AS country FROM `chrome-ux-report.country_in.201806`
    UNION ALL
      SELECT *, 'US' AS country FROM `chrome-ux-report.country_us.201806`)
      
SELECT
    country,
    effective_connection_type.name AS ect,
    SUM(bin.density) AS density
FROM
    countries,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    bin.end <= 5000 AND
    origin = 'https://www.kaggle.com'
GROUP BY
    country,
    ect
ORDER BY
    density DESC;
        """
print(chromeUXreport.estimate_query_size(query4))
response4 = chromeUXreport.query_to_pandas_safe(query4,max_gb_scanned=6)
response4.head(20)
result4 = response4
sns.factorplot(x='country', y='density', hue='ect', data=result4, kind='bar', size=4, aspect=2.0)
query5 = """
SELECT
  ROUND(SUM(IF(fid.start < 100, fid.density, 0)), 4) AS fast_fid
FROM
  `chrome-ux-report.all.201806`,
  UNNEST(experimental.first_input_delay.histogram.bin) AS fid
WHERE
  origin = 'https://www.kaggle.com';
        """
print(chromeUXreport.estimate_query_size(query5))
response5 = chromeUXreport.query_to_pandas_safe(query5,max_gb_scanned=3)
response5.head(20)
query6 = """
SELECT
  ROUND(SUM(IF(fid.start < 100, fid.density, 0)) / SUM(fid.density), 4) AS fast_fid
FROM
  `chrome-ux-report.all.201806`,
  UNNEST(experimental.first_input_delay.histogram.bin) AS fid;
        """
print(chromeUXreport.estimate_query_size(query6))
response6 = chromeUXreport.query_to_pandas_safe(query6,max_gb_scanned=3)
response6.head(20)
query7 = """
SELECT
  form_factor.name AS form_factor,
  ROUND(SUM(IF(fid.start < 100, fid.density, 0)) / SUM(fid.density), 4) AS fast_fid
FROM
  `chrome-ux-report.all.201806`,
  UNNEST(experimental.first_input_delay.histogram.bin) AS fid
WHERE
  origin = 'https://www.kaggle.com'
GROUP BY
  form_factor;
        """
print(chromeUXreport.estimate_query_size(query7))
response7 = chromeUXreport.query_to_pandas_safe(query7,max_gb_scanned=3)
response7.head(20)
result7 = response7
sns.factorplot(x='form_factor', y='fast_fid', data=result7, kind='bar', size=4, aspect=2.0)
query8 = """#standardSQL
SELECT
    origin,
    ROUND(SUM(IF(fcp.start < 1000, fcp.density, 0)) / SUM(fcp.density) * 100) AS fast_fcp
FROM
    `chrome-ux-report.all.201806`,
    UNNEST(first_contentful_paint.histogram.bin) AS fcp
WHERE
    origin IN ('https://www.analyticsvidhya.com', 'https://www.kdnuggets.com','https://medium.com')
GROUP BY
    origin;
        """
print(chromeUXreport.estimate_query_size(query8))
response8 = chromeUXreport.query_to_pandas_safe(query8,max_gb_scanned=5)
response8.head(20)
result8 = response8
sns.factorplot(x='origin', y='fast_fcp', data=result8, kind='bar', size=4, aspect=2.0)
query9 = """#standardSQL
SELECT
  origin,
  ROUND(SUM(IF(bin.start < 1000, bin.density, 0)) / SUM(bin.density), 4) AS fast_fcp,
  ROUND(SUM(IF(bin.start >= 1000 AND bin.start < 3000, bin.density, 0)) / SUM(bin.density), 4) AS avg_fcp,
  ROUND(SUM(IF(bin.start >= 3000, bin.density, 0)) / SUM(bin.density), 4) AS slow_fcp
FROM
    `chrome-ux-report.all.201806`,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    origin IN ('https://www.analyticsvidhya.com', 'https://www.kdnuggets.com','https://medium.com')
GROUP BY
    origin;
        """
print(chromeUXreport.estimate_query_size(query9))
response9 = chromeUXreport.query_to_pandas_safe(query9,max_gb_scanned=5)
response9.head(20)
barWidth = 0.85
r = response9.origin
greenBars = response9.fast_fcp
orangeBars = response9.avg_fcp
blueBars = response9.slow_fcp
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth)
query10 = """#standardSQL
SELECT
    origin,
    ROUND(SUM(IF(form_factor.name = 'desktop', fcp.density, 0)) / SUM(fcp.density) * 100) AS pct_desktop,
    ROUND(SUM(IF(form_factor.name = 'phone', fcp.density, 0)) / SUM(fcp.density) * 100) AS pct_phone,
    ROUND(SUM(IF(form_factor.name = 'tablet', fcp.density, 0)) / SUM(fcp.density) * 100) AS pct_tablet
    
FROM
    `chrome-ux-report.all.201806`,
    UNNEST(first_contentful_paint.histogram.bin) AS fcp
WHERE
    origin IN ('https://www.analyticsvidhya.com', 'https://www.kdnuggets.com','https://medium.com')
GROUP BY
    origin;
        """
print(chromeUXreport.estimate_query_size(query10))
response10 = chromeUXreport.query_to_pandas_safe(query10,max_gb_scanned=3)
response10.head(20)
barWidth = 0.85
r = response10.origin
greenBars = response10.pct_desktop
orangeBars = response10.pct_phone
blueBars = response10.pct_tablet

# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth)
query11 = """#standardSQL
SELECT
  origin,
  effective_connection_type.name AS ect,
  ROUND(SUM(bin.density), 4) AS density
    
FROM
    `chrome-ux-report.all.201806`,
    UNNEST(first_contentful_paint.histogram.bin) AS bin
WHERE
    origin IN ('https://www.analyticsvidhya.com', 'https://www.kdnuggets.com','https://medium.com')
GROUP BY
    origin,
    ect
ORDER BY
    origin,
    ect;
        """
print(chromeUXreport.estimate_query_size(query11))
response11 = chromeUXreport.query_to_pandas_safe(query11,max_gb_scanned=3)
response11.head(20)
result11 = response11
sns.factorplot(x='origin', y='density', hue='ect', data=result11, kind='bar', size=4, aspect=2.0)
query12 = """
SELECT
  origin,
  ROUND(SUM(IF(fid.start < 100, fid.density, 0)), 4) AS fast_fid
FROM
  `chrome-ux-report.all.201806`,
  UNNEST(experimental.first_input_delay.histogram.bin) AS fid
WHERE
  origin IN ('https://www.analyticsvidhya.com', 'https://www.kdnuggets.com','https://medium.com')
GROUP BY
  origin;
        """
print(chromeUXreport.estimate_query_size(query12))
response12 = chromeUXreport.query_to_pandas_safe(query12,max_gb_scanned=3)
response12.head(20)
result12 = response12
sns.factorplot(x='origin', y='fast_fid', data=result12, kind='bar', size=4, aspect=2.0)
