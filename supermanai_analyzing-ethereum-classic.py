# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# With the help of https://www.kaggle.com/yazanator/analyzing-ethereum-classic-via-google-bigquery

from google.cloud import bigquery

!pip install plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
client = bigquery.Client()

ethereum_classic_dataset_ref = client.dataset('crypto_ethereum_classic', project='bigquery-public-data')
query = """

WITH mined_block AS (

  SELECT miner, DATE(timestamp)

  FROM `bigquery-public-data.crypto_ethereum_classic.blocks` 

  WHERE DATE(timestamp) > DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)

  ORDER BY miner ASC)

SELECT miner, COUNT(miner) AS total_block_reward 

FROM mined_block 

GROUP BY miner 

ORDER BY total_block_reward DESC

LIMIT 10

"""



query_job = client.query(query)

iterator = query_job.result()
rows = list(iterator)

# Transform the rows into a nice pandas dataframe

top_miners = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines

top_miners.head(10)