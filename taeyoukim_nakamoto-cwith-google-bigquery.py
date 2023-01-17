import numpy as np

import pandas as pd

import os

from google.cloud import bigquery

# For this Notebook, we will be using the Nakamoto python 

# module found here: https://github.com/YazzyYaz/nakamoto-coefficient

!pip install nakamoto
client = bigquery.Client()

ethereum_dataset_ref = client.dataset('crypto_ethereum_classic', project='bigquery-public-data')
# SQL query needed to get top 10K Ethereum Classic balances for the day

query = """

#standardSQL

-- MIT License

-- Copyright (c) 2018 Evgeny Medvedev, evge.medvedev@gmail.com

with double_entry_book as (

    -- debits

    select to_address as address, value as value

    from `bigquery-public-data.crypto_ethereum_classic.traces`

    where to_address is not null

    and status = 1

    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)

    union all

    -- credits

    select from_address as address, -value as value

    from `bigquery-public-data.crypto_ethereum_classic.traces`

    where from_address is not null

    and status = 1

    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)

    union all

    -- transaction fees debits

    select miner as address, sum(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value

    from `bigquery-public-data.crypto_ethereum_classic.transactions` as transactions

    join `bigquery-public-data.crypto_ethereum_classic.blocks` as blocks on blocks.number = transactions.block_number

    group by blocks.miner

    union all

    -- transaction fees credits

    select from_address as address, -(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value

    from `bigquery-public-data.crypto_ethereum_classic.transactions`

)

select address, 

sum(value) / 1000000000 as balance

from double_entry_book

group by address

order by balance desc

limit 10000

"""



# We pass the query to the client

query_job = client.query(query)

iterator = query_job.result()
rows = list(iterator)

# Transform the rows into a nice pandas dataframe

balances = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
# Now that we have a dataframe of the top 100k balances for the day on ETC,

# it's time to import it into a CustomSector in nakamoto to begin analysis



from nakamoto.sector import CustomSector



# data passed into nakamoto must be as a numpy array

balance_data = np.array(balances['balance'])

type(balance_data)
# We build a config dictionary for Plotly like this:

nakamoto_config = {

    'plot_notebook': True,

    'plot_image_path': None

}



# We also need a currency name and sector type, which is used for plotting information

currency = 'ETC'

sector_type = 'daily balance'



# Since our balance data is sorted descending, we need to flip the data

# for a proper gini and lorenz, otherwise the coefficient comes back negative

balance_data = balance_data[::-1]



# Now, we instantiate the balance object

balance_sector = CustomSector(balance_data,

                             currency,

                             sector_type,

                             **nakamoto_config)
# Let's get back the gini coefficient

balance_sector.get_gini_coefficient()
balance_sector.get_nakamoto_coefficient()
balance_sector.get_plot()
query = """

#standardSQL

-- MIT License

-- Copyright (c) 2019 Yaz Khoury, yaz.khoury@gmail.com

WITH mined_block AS (

  SELECT miner, DATE(timestamp)

  FROM `bigquery-public-data.crypto_ethereum_classic.blocks` 

  WHERE DATE(timestamp) > DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)

  ORDER BY miner ASC)

SELECT miner, COUNT(miner) AS total_block_reward 

FROM mined_block 

GROUP BY miner 

ORDER BY total_block_reward ASC

"""



# We pass the query to the client

query_job = client.query(query)

iterator = query_job.result()
rows = list(iterator)

# Transform the rows into a nice pandas dataframe

mining_rewards = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

mining_rewards_data = np.array(mining_rewards['total_block_reward'])
sector_type = 'mining_rewards'

mining_rewards_sector = CustomSector(mining_rewards_data,

                             currency,

                             sector_type,

                             **nakamoto_config)
# Mining Gini

mining_rewards_sector.get_gini_coefficient()
# Nakamoto Coefficient

mining_rewards_sector.get_nakamoto_coefficient()
mining_rewards_sector.get_plot()
from nakamoto.sector import Market



market_url = "https://coinmarketcap.com/currencies/ethereum-classic/#markets"

market_sector = Market(currency, market_url, **nakamoto_config)
# Market Gini

market_sector.get_gini_coefficient()
market_sector.get_nakamoto_coefficient()
market_sector.get_plot()
from nakamoto.sector import Client, Geography



client_sector = Client(currency, **nakamoto_config)

geography_sector = Geography(currency, **nakamoto_config)
# Client Gini

client_sector.get_gini_coefficient()
# Geography Gini

geography_sector.get_gini_coefficient()
# Client Nakamoto

client_sector.get_nakamoto_coefficient()
# Geography Nakamoto

geography_sector.get_nakamoto_coefficient()
client_sector.get_plot()
geography_sector.get_plot()
from nakamoto.coefficient import Nakamoto





sector_list = [geography_sector, 

               market_sector, 

               client_sector,  

               balance_sector,

               mining_rewards_sector]



nakamoto = Nakamoto(sector_list)
# Minimum Nakamoto Coefficient

nakamoto.get_minimum_nakamoto()
# Maximum Gini Coefficient

nakamoto.get_maximum_gini()