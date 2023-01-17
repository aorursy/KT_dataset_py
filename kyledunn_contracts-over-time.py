import numpy as np

import pandas as pd
from bq_helper import BigQueryHelper



# Helper object for BigQuery Ethereum dataset

eth = BigQueryHelper(active_project="bigquery-public-data", 

                     dataset_name="crypto_ethereum")
query = """

WITH contracts_in_block as ( 

    SELECT

      block_timestamp as timestamp,

      count(*) as count

    FROM

      `bigquery-public-data.crypto_ethereum.transactions`

    WHERE

      to_address is null -- contract creation indicator

    GROUP BY

      1

)



SELECT

    *

FROM

    contracts_in_block

ORDER BY

    1 asc

"""
# Estimate total data scanned for query (in GB)

eth.estimate_query_size(query)
# Store the results into a Pandas DataFrame

df = eth.query_to_pandas_safe(query, max_gb_scanned=20)
# Convert strings into datetime objects and resample the data to monthly totals



df['date'] = pd.to_datetime(df.timestamp)

agg = df.resample('M', on='date').sum()
agg.plot(y='count', title='New Ethereum Smart Contracts (monthly totals)', figsize=(16, 9))
# These are better looking charts but they don't persist 

# after nbconvert step and are limited to 5k samples



"""

import altair as alt

alt.renderers.enable('notebook');

""";
"""

alt.Chart(agg.reset_index()).mark_line().encode(

    x='date:T',

    y='count:Q'

).properties(

    title="New Ethereum Smart Contracts by Month",

    height=400,

    width=750

)

""";
# Trying to save the Altair/Vega chart as PNG fails - Selenium is required but unavailable



# from io import StringIO

# temp = StringIO()

# newByMonth.save(temp, format='png')
"""

alt.Chart(agg.reset_index()).mark_line().encode(

    x='date:T',

    y=alt.Y('count:Q', scale=alt.Scale(type='log', base=10))

).properties(

    title="New Ethereum Smart Contracts [log10] by Month",

    height=400,

    width=750

)

""";
# Byzantium: > 4370000



rtx_query = """

WITH reverted_transactions_in_block as ( 

    SELECT

      block_timestamp as timestamp,

      count(*) as count

    FROM

      `bigquery-public-data.crypto_ethereum.transactions`

    WHERE

      block_number > 4370000 -- first Byzantium block

      AND

      receipt_status = 0 -- reverted transaction indicator

    GROUP BY

      1

)



SELECT

    *

FROM

    reverted_transactions_in_block

ORDER BY

    1 asc

"""
# Store the results into a Pandas DataFrame

df_rtx = eth.query_to_pandas_safe(rtx_query, max_gb_scanned=22)
df_rtx['date'] = pd.to_datetime(df_rtx.timestamp)

agg_rtx = df_rtx.resample('M', on='date').sum()



"""

alt.Chart(agg_rtx.reset_index()).mark_line().encode(

    x='date:T',

    y='count:Q'

).properties(

    title="Reverted Ethereum transactions by Month",

    height=400,

    width=750

)

""";



agg_rtx.plot(y='count', title='Reverted Transactions on Ethereum (monthly totals)', figsize=(16, 9))
rtx_query2 = """

WITH distinct_methods_in_reverted_txns_in_block as ( 

    SELECT

      DATE_TRUNC(CAST(block_timestamp as date), MONTH) as timestamp,

      count(distinct(SUBSTR(input, 10))) as unique_methods

    FROM

      `bigquery-public-data.crypto_ethereum.transactions`

    WHERE

      block_number > 4370000 -- first Byzantium block

      AND

      receipt_status = 0 -- reverted transaction indicator

    GROUP BY

      1

)



SELECT

    *

FROM

    distinct_methods_in_reverted_txns_in_block

ORDER BY

    1 asc

"""



# Store the results into a Pandas DataFrame

df_rtx2 = eth.query_to_pandas_safe(rtx_query2, max_gb_scanned=75)
df_rtx2['date'] = pd.to_datetime(df_rtx2.timestamp)



"""

alt.Chart(df_rtx2[['date', 'unique_methods']]).mark_line().encode(

    x='date:T',

    y='unique_methods:Q'

).properties(

    title="Distinct methods called in reverted Ethereum transactions by month",

    height=400,

    width=750

)

""";



df_rtx2.plot(y='unique_methods',

             title='Distinct methods called in Reverted Transactions on Ethereum (monthly totals)',

             figsize=(16, 9))
query2 = """

WITH distinct_methods_in_block as ( 

    SELECT

      DATE_TRUNC(CAST(block_timestamp as date), MONTH) as timestamp,

      count(distinct(SUBSTR(input, 10))) as unique_methods

    FROM

      `bigquery-public-data.crypto_ethereum.transactions`

    GROUP BY

      1

)



SELECT

    *

FROM

    distinct_methods_in_block

ORDER BY

    1 asc

"""



# Store the results into a Pandas DataFrame

df2 = eth.query_to_pandas_safe(query2, max_gb_scanned=75)
df2['date'] = pd.to_datetime(df2.timestamp)

#agg2 = df2[['unique_methods', 'date']].resample('M', on='date').agg(pd.Series.nunique)



"""

alt.Chart(df2[['unique_methods', 'date']]).mark_line().encode(

    x='date:T',

    y='unique_methods:Q'

).properties(

    title="Distinct methods called in Ethereum transactions by month",

    height=400,

    width=750

)

""";



df2.plot(y='unique_methods', title='Distinct methods called on Ethereum', figsize=(16, 9))
"""

alt.Chart(df2[['unique_methods', 'date']]).mark_line().encode(

    x='date:T',

    y=alt.Y('unique_methods:Q', scale=alt.Scale(type='log', base=10))

).properties(

    title="Distinct methods [log10] called in Ethereum transactions by month",

    height=400,

    width=750

)

""";