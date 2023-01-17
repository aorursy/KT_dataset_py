import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper

# Visualizations
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
bq = BigQueryHelper('bigquery-public-data', 'github_repos')
QUERY = """
        SELECT license, COUNT(license) as license_count
        FROM `bigquery-public-data.github_repos.licenses`
        GROUP BY license
        """

df = bq.query_to_pandas_safe(QUERY)
df = df.sort_values('license_count', ascending=False)
data = [
    go.Bar(
        x=df['license'], 
        y=df['license_count'],
        name=df['license'].tolist()
    )
]

layout = dict(
    title = 'Distribution of GitHub Licenses',
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='pandas-bar-chart')
