from google.cloud import bigquery

import pandas as pd



# query ethereum, excluding known exceptions

query = """

select address, symbol,name

from `bigquery-public-data.crypto_ethereum.tokens`

where

  ( upper(name) like '%REIWA%' or upper(symbol) like '%REIWA%' or name like '%令和%' or symbol like '%令和%'

or upper(name) like '%HEISEI%' or upper(symbol) like '%HEISEI%' or name like '%平成%' or symbol like '%平成%'

or upper(name) like '%SHOWA%' or upper(symbol) like '%SHOWA%' or name like '%昭和%' or symbol like '%昭和%'

or upper(name) like '%TAISHO%' or upper(symbol) like '%TAISHO%' or name like '%大正%' or symbol like '%大正%'

or upper(name) like '%MEIJI%' or upper(symbol) like '%MEIJI%' or name like '%明治%' or symbol like '%明治%'

) and upper(symbol) not in ('MJX', 'MJIU', 'MJT')

"""



query_job = bigquery.Client().query(query)

df = query_job.to_dataframe()

from IPython.core.display import HTML

df["address_url"] = df['address'].apply(

    lambda addr: f"<a href='https://etherscan.io/token/{addr}' target='_blank'>Etherscan</a>")

pd.set_option('display.max_colwidth', -1)

HTML(df.to_html(escape=False))