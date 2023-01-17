import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\Python\Python37\d074645e5d17.json"
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

import pandas as pd

#from google.cloud import bigquery - if you use this no need call below one, it's return row wise. But we need DATAFRAME

from pandas.io import gbq # to communicate with Google BigQuery



#Use for offline use of jupyter notebook

init_notebook_mode(connected=True)



#Dynamically load current date

dat=pd.to_datetime('today')

today=dat.strftime("%m-%d-%Y")

#Query 

query = """

WITH double_entry_book AS (

    -- debits

    SELECT

      ARRAY_TO_STRING(inputs.addresses, ",") AS address

    , inputs.type

    , -inputs.value AS value

    FROM `bigquery-public-data.crypto_bitcoin.transactions` JOIN UNNEST(inputs) AS inputs

    WHERE block_timestamp_month = '2019-03-01'

    

   UNION ALL

  -- credits

    SELECT

      ARRAY_TO_STRING(outputs.addresses, ",") AS address

    , outputs.type

    , outputs.value AS value

    FROM `bigquery-public-data.crypto_bitcoin.transactions` JOIN UNNEST(outputs) AS outputs

    WHERE block_timestamp_month = '2019-03-01'

)

SELECT 

  address 

, type

, SUM(value) AS net_change

FROM double_entry_book

GROUP BY 1,2

ORDER BY net_change DESC

LIMIT 10

"""



#Excute query & return to DataFrame. Please Note: Always use Standard SQL Query using [dialect='standard']

tld_share_df = gbq.read_gbq(query, project_id='rahul-data-science',dialect='standard')



#Here we set both axis with data

trace = go.Bar(

    x = tld_share_df['address'],

    y = tld_share_df['net_change']

)



data = [trace]



layout = go.Layout(

    title='Top 10 Richest Bitcoin Holder till '+today

)



#Here we create figure for plot in plotly with above mention 'data' & 'layout'

fig = go.Figure(data=data, layout=layout)



#Here we plot plotly chart diagram in offline manner

py.offline.iplot(fig)