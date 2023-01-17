import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\Python\Python37\d074645e5d17.json"

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

import plotly.figure_factory as ff



import pandas as pd

from pandas.io import gbq # to communicate with Google BigQuery



init_notebook_mode(connected=True)



query = """

SELECT

  EXTRACT(DATE FROM TIMESTAMP_MILLIS(timestamp)) AS date,transaction_id,block_id,previous_block,nonce

  FROM

    `bigquery-public-data.bitcoin_blockchain.transactions`

    LIMIT 1000

"""



tld_share_df = gbq.read_gbq(query, project_id='rahul-data-science',dialect='standard')

trace = go.Scatter(

    x = tld_share_df['transaction_id'],

    y = tld_share_df['date'],

    mode = 'markers'

)



data = [trace]



layout = go.Layout(

    title='bitcoin chart'

)



fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)