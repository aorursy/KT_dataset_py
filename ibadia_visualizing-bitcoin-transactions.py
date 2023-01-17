import time
start_time=time.time()
import numpy as np
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize']=(12,5)
from google.cloud import bigquery
from bq_helper import BigQueryHelper
client = bigquery.Client()
bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")
# Modified the query on the below kernel to get sum of all satoshis spent each day 
# https://www.kaggle.com/mrisdal/visualizing-daily-bitcoin-recipients
q = """
SELECT
  o.day,
  SUM(o.output_price) AS sum_output_price
FROM (
  SELECT
    TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day,
    output.output_satoshis AS output_price
  FROM
    `bigquery-public-data.bitcoin_blockchain.transactions`,
    UNNEST(outputs) AS output ) AS o
GROUP BY
  o.day
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

Grouped_df=bq_assistant.query_to_pandas_safe(q, max_gb_scanned=10)
Grouped_df=Grouped_df.sort_values(by=["sum_output_price"], ascending=False)
#converting satoshis to Bitcoins
Grouped_df["sum_output_price"]=Grouped_df["sum_output_price"].apply(lambda x: float(x)*float(0.00000001))
Grouped_df_ts=Grouped_df.sort_values(by=["day"])
data = [go.Scatter(
            x=Grouped_df_ts["day"],
            y=Grouped_df_ts["sum_output_price"])]
fig = go.Figure(data=data)
print ("_"*30+ " Time series of bitcoin transacted"+"_"*30)
py.offline.iplot(fig)

layout = dict(
    title = "Timeseries from 20th Jan 2016 to 26th Jan 2016",
    xaxis = dict(
        range = ['2016-1-20','2016-1-26'])
)
fig = dict(data=data, layout=layout)
py.offline.iplot(fig)
top=Grouped_df[:5]
top["day"]=top["day"].apply(lambda x: str(x)[:11])
top.plot( x="day",kind="barh")
bottom=Grouped_df[::-1][:5]
bottom["day"]=bottom["day"].apply(lambda x: str(x)[:11])
bottom.plot(kind="barh", x="day")
end_time=time.time()
print ("TOTAL TIME TAKEN FOR KERNEL TO RUN IS :"+ str(end_time-start_time)+" s")