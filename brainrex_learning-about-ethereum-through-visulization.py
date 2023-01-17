# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Here we import the packages we need for the analysis, they are installed in Kaggle's backend 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

!pip install plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
# Connects to BigQuery API through Kaggle, no authentification required from Kaggle.

from google.cloud import bigquery

import pandas as pd



client = bigquery.Client()
# Query by Allen Day, GooglCloud Developer Advocate (https://medium.com/@allenday)

query = """

SELECT 

  SUM(value/POWER(10,18)) AS sum_tx_ether,

  AVG(gas_price*(receipt_gas_used/POWER(10,18))) AS avg_tx_gas_cost,

  DATE(timestamp) AS tx_date

FROM

  `bigquery-public-data.crypto_ethereum.transactions` AS transactions,

  `bigquery-public-data.crypto_ethereum.blocks` AS blocks

WHERE TRUE

  AND transactions.block_number = blocks.number

  AND receipt_status = 1

  AND value > 0

GROUP BY tx_date

HAVING tx_date >= '2017-01-01' AND tx_date <= '2019-05-04'

ORDER BY tx_date

"""
query_job = client.query(query)



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the first 10

df.tail(10)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})



f, g = plt.subplots(figsize=(12, 9))

g = sns.lineplot(x="tx_date", y="avg_tx_gas_cost", data=df, palette="Blues_d")

plt.title("Average Ether transaction cost over time")

plt.show(g)
client = bigquery.Client()

ethereum_classic_dataset_ref = client.dataset('crypto_ethereum', project='bigquery-public-data')




query = """

WITH mined_block AS (

  SELECT miner, DATE(timestamp)

  FROM `bigquery-public-data.crypto_ethereum.blocks` 

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


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go



labels = top_miners['miner']

values = top_miners['total_block_reward']



trace = go.Pie(labels=labels, values=values)



iplot([trace])



query = """

#standardSQL

-- MIT License

-- Copyright (c) 2019 Yaz Khoury, yaz.khoury@gmail.com



SELECT miner, 

    DATE(timestamp) as date,

    COUNT(miner) as total_block_reward

FROM `bigquery-public-data.crypto_ethereum.blocks` 

GROUP BY miner, date

HAVING COUNT(miner) > f

ORDER BY date, COUNT(miner) ASC

"""

query_job = client.query(query)

iterator = query_job.result()

rows = list(iterator)

# Transform the rows into a nice pandas dataframe

top_miners_by_date = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

top_miners_by_date.head(10)



date_series = top_miners_by_date['date'].unique()

date_series



traces = []

miner_series = top_miners_by_date['miner'].unique()



for index, miner in enumerate(miner_series):

    miner_reward_by_date = top_miners_by_date.loc[top_miners_by_date['miner'] == miner]

    miner_reward = miner_reward_by_date['total_block_reward']

    miner_date = miner_reward_by_date['date']

    trace = dict(

        x=miner_date,

        y=miner_reward,

        mode='lines',

        stackgroup='one'

    )

    traces.append(trace)

fig = dict(data=traces)



iplot(fig)
query = """

#standardSQL

-- MIT License

-- Copyright (c) 2019 Yaz Khoury, yaz.khoury@gmail.com



SELECT miner, 

    DATE(timestamp) as date,

    COUNT(miner) as total_block_reward

FROM `bigquery-public-data.crypto_ethereum_classic.blocks` 

GROUP BY miner, date

HAVING COUNT(miner) > f

ORDER BY date, COUNT(miner) ASC

"""

query_job = client.query(query)

iterator = query_job.result()

query = """

#standardSQL

-- MIT License

-- Copyright (c) 2019 Yaz Khoury, yaz.khoury@gmail.com



WITH total_reward_book AS (

  SELECT miner, 

    DATE(timestamp) as date,

    COUNT(miner) as total_block_reward

  FROM `bigquery-public-data.crypto_ethereum.blocks` 

  GROUP BY miner, date

  HAVING COUNT(miner) > 100

),

total_reward_book_by_date AS (

 SELECT date, 

        miner AS address, 

        SUM(total_block_reward / POWER(10,0)) AS value

  FROM total_reward_book

  GROUP BY miner, date

),

daily_rewards_with_gaps AS (

  SELECT

    address, 

    date,

    SUM(value) OVER (PARTITION BY ADDRESS ORDER BY date) AS block_rewards,

    LEAD(date, 1, CURRENT_DATE()) OVER (PARTITION BY ADDRESS ORDER BY date) AS next_date

  FROM total_reward_book_by_date

),

calendar AS (

  SELECT date 

  FROM UNNEST(GENERATE_DATE_ARRAY('2015-07-30', CURRENT_DATE())) AS date

),

daily_rewards AS (

  SELECT address, 

    calendar.date, 

    block_rewards

  FROM daily_rewards_with_gaps

  JOIN calendar ON daily_rewards_with_gaps.date <= calendar.date 

  AND calendar.date < daily_rewards_with_gaps.next_date

),

supply AS (

  SELECT date,

    SUM(block_rewards) AS total_rewards

  FROM daily_rewards

  GROUP BY date

),

ranked_daily_rewards AS (

  SELECT daily_rewards.date AS date,

    block_rewards,

    ROW_NUMBER() OVER (PARTITION BY daily_rewards.date ORDER BY block_rewards DESC) AS rank

  FROM daily_rewards

  JOIN supply ON daily_rewards.date = supply.date

  WHERE SAFE_DIVIDE(block_rewards, total_rewards) >= 0.01

  ORDER BY block_rewards DESC

),

daily_gini AS (

  SELECT date,

    -- (1 − 2B) https://en.wikipedia.org/wiki/Gini_coefficient

    1 - 2 * SUM((block_rewards * (rank - 1) + block_rewards / 2)) / COUNT(*) / SUM(block_rewards) AS gini

  FROM ranked_daily_rewards

  GROUP BY DATE

)

SELECT date,

  gini,

  AVG(gini) OVER (ORDER BY date ASC ROWS 7 PRECEDING) AS gini_sma_7,

  AVG(gini) OVER (ORDER BY date ASC ROWS 30 PRECEDING) AS gini_sma_30

FROM daily_gini

ORDER BY date ASC

"""



query_job = client.query(query)

iterator = query_job.result()
rows = list(iterator)

# Transform the rows into a nice pandas dataframe

mining_reward_gini_by_date = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

mining_reward_gini_by_date.head(10)

traces = []

x = mining_reward_gini_by_date['date']

gini_list = ['gini', 'gini_sma_7', 'gini_sma_30']

for gini in gini_list:

    y = mining_reward_gini_by_date[gini]

    trace = dict(

        x=x,

        y=y,

        

        mode='lines'

    )

    traces.append(trace)

fig = dict(data=traces)



iplot(fig, validate=False)

query = """

with double_entry_book as (

    -- debits

    select to_address as address, value as value

    from `bigquery-public-data.crypto_ethereum.traces`

    where to_address is not null

    and status = 1

    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)

    union all

    -- credits

    select from_address as address, -value as value

    from `bigquery-public-data.crypto_ethereum.traces`

    where from_address is not null

    and status = 1

    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)

    union all

    -- transaction fees debits

    select miner as address, sum(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value

    from `bigquery-public-data.crypto_ethereum.transactions` as transactions

    join `bigquery-public-data.crypto_ethereum.blocks` as blocks on blocks.number = transactions.block_number

    group by blocks.miner

    union all

    -- transaction fees credits

    select from_address as address, -(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value

    from `bigquery-public-data.crypto_ethereum.transactions`

)

select address, 

sum(value) / 1000000000 as balance

from double_entry_book

group by address

order by balance desc

limit 20

"""



query_job = client.query(query)

iterator = query_job.result()

rows = list(iterator)

# Transform the rows into a nice pandas dataframe

top_address_rich_list = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

top_address_rich_list.head(10)

labels = top_address_rich_list['address']

values = top_address_rich_list['balance']



trace = go.Pie(labels=labels, values=values)



iplot([trace])





query = """

with 

double_entry_book as (

    -- debits

    select to_address as address, value as value, block_timestamp

    from `bigquery-public-data.crypto_ethereum.traces`

    where to_address is not null

    and status = 1

    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)

    union all

    -- credits

    select from_address as address, -value as value, block_timestamp

    from `bigquery-public-data.crypto_ethereum.traces`

    where from_address is not null

    and status = 1

    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)

    union all

    -- transaction fees debits

    select miner as address, sum(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value, block_timestamp

    from `bigquery-public-data.crypto_ethereum.transactions` as transactions

    join `bigquery-public-data.crypto_ethereum.blocks` as blocks on blocks.number = transactions.block_number

    group by blocks.miner, block_timestamp

    union all

    -- transaction fees credits

    select from_address as address, -(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value, block_timestamp

    from `bigquery-public-data.crypto_ethereum.transactions`

),

double_entry_book_by_date as (

    select 

        date(block_timestamp) as date, 

        address, 

        sum(value / POWER(10,0)) as value

    from double_entry_book

    group by address, date

),

daily_balances_with_gaps as (

    select 

        address, 

        date,

        sum(value) over (partition by address order by date) as balance,

        lead(date, 1, current_date()) over (partition by address order by date) as next_date

        from double_entry_book_by_date

),

calendar as (

    select date from unnest(generate_date_array('2015-07-30', current_date())) as date

),

daily_balances as (

    select address, calendar.date, balance

    from daily_balances_with_gaps

    join calendar on daily_balances_with_gaps.date <= calendar.date and calendar.date < daily_balances_with_gaps.next_date

),

 supply as (

    select

        date,

        sum(balance) as daily_supply

    from daily_balances

    group by date

),

ranked_daily_balances as (

    select 

        daily_balances.date,

        balance,

        row_number() over (partition by daily_balances.date order by balance desc) as rank

    from daily_balances

    join supply on daily_balances.date = supply.date

    where safe_divide(balance, daily_supply) >= 0.0001

    ORDER BY safe_divide(balance, daily_supply) DESC

), 

gini_daily as (

   select

    date,

    -- (1 − 2B) https://en.wikipedia.org/wiki/Gini_coefficient

    1 - 2 * sum((balance * (rank - 1) + balance / 2)) / count(*) / sum(balance) as gini

  from ranked_daily_balances

  group by date

)

select date,

    gini,

    avg(gini) over (order by date asc rows 7 preceding) as gini_sma7,

    avg(gini) over (order by date asc rows 30 preceding) as gini_sma30

from gini_daily

order by date asc

"""



query_job = client.query(query)

iterator = query_job.result()
rows = list(iterator)

# Transform the rows into a nice pandas dataframe

daily_balance_gini = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

daily_balance_gini.head(10)

traces = []

x = daily_balance_gini['date']

gini_list = ['gini', 'gini_sma7', 'gini_sma30']

for gini in gini_list:

    y = daily_balance_gini[gini]

    trace = dict(

        x=x,

        y=y,

        mode='lines'

    )

    traces.append(trace)

fig = dict(data=traces)



iplot(fig, validate=False)
query = """

#standardSQL

-- MIT License

-- Copyright (c) 2019 Yaz Khoury, yaz.khoury@gmail.com



WITH block_rows AS (

  SELECT *, ROW_NUMBER() OVER (ORDER BY timestamp) AS rn

  FROM `bigquery-public-data.crypto_ethereum.blocks`

),

delta_time AS (

  SELECT

  mp.timestamp AS block_time,

  mp.difficulty AS difficulty,

  TIMESTAMP_DIFF(mp.timestamp, mc.timestamp, SECOND) AS delta_block_time

  FROM block_rows mc

  JOIN block_rows mp

  ON mc.rn = mp.rn - 1

),

hashrate_book AS (

  SELECT TIMESTAMP_TRUNC(block_time, DAY) AS block_day,

  AVG(delta_block_time) as daily_avg_block_time,

  AVG(difficulty) as daily_avg_difficulty

  FROM delta_time

  GROUP BY TIMESTAMP_TRUNC(block_time, DAY)

)

SELECT block_day,

(daily_avg_difficulty/daily_avg_block_time)/1000000000 as hashrate

FROM hashrate_book

ORDER BY block_day ASC

"""



query_job = client.query(query)

iterator = query_job.result()

rows = list(iterator)

# Transform the rows into a nice pandas dataframe

daily_hashrate = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

daily_hashrate.head(10)

trace = go.Scatter(

    x=daily_hashrate['block_day'],

    y=daily_hashrate['hashrate'],

    mode='lines'

)

data = [trace]

iplot(data)



train = pd.read_csv('..input/etherclose/eth-24.csv')