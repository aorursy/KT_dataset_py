from google.cloud import bigquery

from bq_helper import BigQueryHelper

import pandas as pd



# This establishes an authenticated session and prepares a reference to the dataset that lives in BigQuery.

bq_assistant = BigQueryHelper("bigquery-public-data", "ethereum_blockchain")



# This initiates a google big query client without a reference to a specific dataset

client = bigquery.Client()
bq_assistant.list_tables()
# Quering GBQ with BigQueryHelper

    

    # BigQueryHelper Functions

# Queries dataset and returns a panda dataframe

##df = bq_assistant.query_to_pandas(query)



# Queries dataset and returns a panda dataframe + allows to set a max scan limit

##df = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=40)



# Lists all tables in the dataset

##bq_assistant.list_tables()



# Shows the head of a specific table

##bq_assistant.head("table_name", num_rows=3)



# Shows details about colums 

##bq_assistant.table_schema("table_name")



# check estimated size of a query

##bq_assistant.estimate_query_size(query)



    # other usefull functions

# Print size of dataframe

##print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))

# Quering GBQ at Kaggle with GBQ client (without BigQueryHelper)

##query_job = client.query(query)



##iterator = query_job.result(timeout=30)

##rows = list(iterator)



# Transform the rows into a nice pandas dataframe

##df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the first 10

##df.head(10)
# Query examples



min_block_number = 5100000

max_block_number = 5200000   #usually 6400000



# select distinct transaction senders within a given range of blocks 

query_1 = """

SELECT DISTINCT

    from_address AS sender, block_number AS block_number

FROM

    `bigquery-public-data.ethereum_blockchain.transactions`

WHERE

    block_number > %d

    AND

    block_number < %d

"""



# select distinct transaction senders within a given range of blocks

query_2 = """

SELECT DISTINCT

    to_address AS receipient, block_number AS block_number

FROM

    `bigquery-public-data.ethereum_blockchain.transactions`

WHERE

    block_number > %d

    AND

    block_number < %d

"""



# select transaction senders that are contracts 

query_3 = """

SELECT

    DISTINCT address AS sc_address, is_erc20, is_erc721, block_number AS block_number

FROM

    `bigquery-public-data.ethereum_blockchain.contracts`

WHERE

    block_number > %d

    AND

    block_number < %d

"""
# query_1: unique senders

unique_senders = bq_assistant.query_to_pandas_safe(query_1 % (min_block_number, max_block_number), max_gb_scanned=52)

print("Retrieved " + str(len(unique_senders)) + " unique_senders.")

print(unique_senders.head(10))
# query_2: unique recipients

unique_receipients = bq_assistant.query_to_pandas_safe(query_2 % (min_block_number, max_block_number), max_gb_scanned=52)

print("Retrieved " + str(len(unique_receipients)) + " unique_receipients.")

print(unique_receipients.head(10))
# query_3: unique contracts

unique_contracts = bq_assistant.query_to_pandas_safe(query_3 % (min_block_number, max_block_number), max_gb_scanned=52)

print("Retrieved " + str(len(unique_contracts)) + " unique_contracts.")

print(unique_contracts.head(10))
from google.cloud import bigquery

from bq_helper import BigQueryHelper

import pandas as pd



bq_assistant = BigQueryHelper("bigquery-public-data", "ethereum_blockchain")

client = bigquery.Client()



min_block_number = 5100000

max_block_number = 6400000



# find average values and sort

query = """

SELECT

  address, SUM(n_updates) AS updates

FROM

(

  SELECT

      address, COUNT(*) AS n_updates

  FROM

  (

  SELECT DISTINCT

    from_address AS address, block_number AS block_number

  FROM

    `bigquery-public-data.ethereum_blockchain.transactions`

  WHERE

    block_number > %d

    AND

    block_number < %d

  )

  GROUP BY 

    address



  UNION ALL



  SELECT 

      address AS address, COUNT(*) AS n_updates

  FROM

  (

  SELECT DISTINCT

    to_address AS address, block_number AS block_number

  FROM

    `bigquery-public-data.ethereum_blockchain.transactions`

  WHERE

    block_number > %d

    AND

    block_number < %d

  )

  GROUP BY 

    address

)

WHERE

  n_updates >= 5

  AND

  address IS NOT NULL

GROUP BY 

  address

ORDER BY 

  updates DESC

"""



most_populars = bq_assistant.query_to_pandas_safe(query % (min_block_number, max_block_number, min_block_number, max_block_number), max_gb_scanned=100)

print("Retrieved " + str(len(most_populars)) + " accounts.")

blocks_int = max_block_number - min_block_number

most_populars = most_populars.sort_values(by='updates', ascending=False)

most_populars["probability"] = most_populars["updates"] / (blocks_int*1.0)

print(most_populars.head(10))
from scipy.optimize import curve_fit

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import numpy as np



def func_powerlaw(x, m, c, c0):

    return c0 + x**m * c



blocks_int = max_block_number - min_block_number



# Compute probabilities

most_populars["probability"] = most_populars["updates"] / (blocks_int*1.0)

most_populars["idxs"] = range(1, len(most_populars) + 1)



# Fit curve

sol = curve_fit(func_powerlaw, most_populars["idxs"], most_populars["probability"], p0 = np.asarray([float(-1),float(10**5),0]))

fitted_func = func_powerlaw(most_populars["idxs"], sol[0][0], sol[0][1], sol[0][2])

print("Fit with values {} {} {}".format(sol[0][0], sol[0][1], sol[0][2]))



# Plot fit vs samples (only for the first 2000)

plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(10,5))

plt.loglog(most_populars["probability"].tolist()[1:10000],'o')

plt.loglog(fitted_func.tolist()[1:10000])

plt.xlabel("Account index (by descending popularity)")

plt.ylabel("Relative frequency [1/block]")

plt.show()
import matplotlib.pyplot as plt

import numpy as np



query = """

SELECT 

      timestamp, number

    FROM

      `bigquery-public-data.ethereum_blockchain.blocks`

INNER JOIN 

(

    SELECT DISTINCT

              from_address AS address, block_number AS block_number

            FROM

              `bigquery-public-data.ethereum_blockchain.transactions`

            WHERE

              from_address = '%s'

              AND

              block_number > %d

              AND

              block_number < %d



    UNION DISTINCT



    SELECT DISTINCT

              to_address AS address, block_number AS block_number

            FROM

              `bigquery-public-data.ethereum_blockchain.transactions`

            WHERE

              to_address = '%s'

              AND

              block_number > %d

              AND

              block_number < %d

) as InnerTable

ON 

    `bigquery-public-data.ethereum_blockchain.blocks`.number = InnerTable.block_number;

"""



# CryptoKitties address

adx_1 = most_populars.iloc[4].address 

transax_1 = bq_assistant.query_to_pandas_safe(query % (adx_1, min_block_number, max_block_number, adx_1, min_block_number, max_block_number), max_gb_scanned=60)

print("Retrieved " + str(len(transax_1)) + " blocks for account %s." % (adx_1) )

transax_1.sort_values(by="number", ascending=True, inplace=True)

    

# Bittrex address    

adx_2 = most_populars.iloc[3].address 

transax_2 = bq_assistant.query_to_pandas_safe(query % (adx_2, min_block_number, max_block_number, adx_2, min_block_number, max_block_number), max_gb_scanned=60)

print("Retrieved " + str(len(transax_2)) + " blocks for account %s." % (adx_2) )

transax_2.sort_values(by="number", ascending=True, inplace=True)

    

transax = list()

transax.append(transax_1)

transax.append(transax_2)
# plot the Empirical CDF



plt.figure(figsize=(15,5))

for t in transax:

    t.sort_values(by="number", inplace=True)

    tx_d = t.diff()

    tx_d = tx_d.iloc[1:]

    count = np.sort(tx_d["number"].values)

    cdf = np.arange(len(count)+1)/float(len(count))

    plt.plot(count, cdf[:-1])



plt.axis([0, 20, 0, 1])

plt.xlabel("Number of blocks without updates [n]")

plt.ylabel("Empirical CDF ( Pr[x <= n] )")

plt.show()
# activity during time

txp_1 = transax_1["timestamp"].groupby(transax_1["timestamp"].dt.floor('d')).size().reset_index(name='CryptoKitties')

txp_2 = transax_2["timestamp"].groupby(transax_2["timestamp"].dt.floor('d')).size().reset_index(name='Bittrex')

txp_1 = txp_1[2:-2]

txp_2 = txp_2[2:-2]



fig, ax = plt.subplots(1, 1, figsize=(15, 8))

ax = txp_1.plot(x="timestamp", y="CryptoKitties", ax=ax)

ax = txp_2.plot(x="timestamp", y="Bittrex", ax=ax)

plt.ylabel("Active blocks/day")



# patterns

f = plt.figure(figsize=(15,5))

ax = f.add_subplot(121)

ax2 = f.add_subplot(122)



plt.subplot(1, 2, 1)

txp_1 = transax_1["timestamp"].groupby(transax_1["timestamp"].dt.day_name()).count().sort_values()

txp_1 /= sum(txp_1)

txp_1.plot(kind="bar", ax=ax)

plt.xlabel("Day of the week")

plt.ylabel("Normalized count")

plt.title("CryptoKitties")



plt.subplot(1, 2, 2)

txp_2 = transax_2["timestamp"].groupby(transax_2["timestamp"].dt.day_name()).count().sort_values()

txp_2 /= sum(txp_2)

txp_2.plot(kind="bar", ax=ax2)

plt.xlabel("Day of the week")

plt.ylabel("Normalized count")

plt.title("Bittrex")
# 0 Getting started

from google.cloud import bigquery

from bq_helper import BigQueryHelper

import pandas as pd



bq_assistant = BigQueryHelper("bigquery-public-data", "ethereum_blockchain")
# 1.1 Query for unique deployers ordered by contract count including first and last creation timestamps

query_deployers = """

SELECT

    DISTINCT from_address AS unique_deployer,

    COUNT(1) AS contracts_deployed,

    MIN(block_timestamp) AS first_event,

    MAX(block_timestamp) AS last_event,

    DATE_DIFF(DATE(MAX(block_timestamp)),DATE(MIN(block_timestamp)),day) AS days_active

FROM   `bigquery-public-data.ethereum_blockchain.transactions`

        WHERE

        

        block_number < 6400000 

        AND

        block_number > 5100000 

        AND

        to_address IS null

        AND

        receipt_status = 1

    GROUP BY

        from_address

    HAVING

        COUNT(1) > 4

    ORDER BY

        contracts_deployed DESC

"""



deployers = bq_assistant.query_to_pandas_safe(query_deployers, max_gb_scanned=100)

deployers.describe()
# 1.1 continued: Summary of unique deployers ordered by number of deployed contracts (limited to 95th percentile)

deployers2 = deployers[deployers.contracts_deployed < deployers.contracts_deployed.quantile(.95)]

deployers2.describe()
# 1.1 continued: Graph for unique deployers ordered by number of deployed contracts (limited to 95th percentile)

import matplotlib.pyplot as plt



fig,ax = plt.subplots()

plt.plot(deployers2.contracts_deployed)

plt.title("Contracts created per deployer \n limited to the 95th percentile", y=1.01, fontsize=20)

plt.ylabel("Contracts created", labelpad=15)

plt.xlabel("Unique deployers", labelpad=15)

plt.legend()

# 1.2 Querying unique vs duplicate bytecodes in contract creation

query_dupl_bytecode = """

SELECT DISTINCT creator, COUNT(occurrence) as duplicates_creator, SUM(occurrence) AS duplicates, SUM(uniques) AS uniques

FROM (

  SELECT

    DISTINCT(output) AS bytecode,

    from_address,

    CASE

      WHEN trace_address IS NOT NULL THEN 'contract'

    ELSE

    'user'

  END

    AS creator,

    CASE

      WHEN COUNT(*) > 1 THEN COUNT(*)

  END

    AS occurrence,

    CASE

      WHEN COUNT(*) = 1 THEN 1

  END

    AS uniques

  FROM

    `bigquery-public-data.crypto_ethereum.traces`

  WHERE

    trace_type = 'create'

    AND status = 1

    AND block_number > 5400000

    AND block_number < 6100000

  GROUP BY

    from_address,

    bytecode,

    creator)

    GROUP BY creator

"""



dupl_bytecode = bq_assistant.query_to_pandas_safe(query_dupl_bytecode, max_gb_scanned=500)

print(dupl_bytecode)
# 1.3 Querying user-created contract deployments over time

import matplotlib.pyplot as plt



query_contract_growth = """

WITH

  a AS (

  SELECT

    DATE(block_timestamp) AS date,

    COUNT(*) AS contracts_creation

  FROM

    `bigquery-public-data.crypto_ethereum.traces` AS traces

  WHERE

    trace_type = 'create'

    AND trace_address IS NULL

  GROUP BY

    date),

  b AS (

  SELECT

    date,

    SUM(contracts_creation) OVER (ORDER BY date) AS ccc,

    LEAD(date, 1) OVER (ORDER BY date) AS next_date

  FROM

    a

  ORDER BY

    date),

  calendar AS (

  SELECT

    date

  FROM

    UNNEST(GENERATE_DATE_ARRAY('2015-07-30', CURRENT_DATE())) AS date),

  c AS (

  SELECT

    calendar.date,

    ccc

  FROM

    b

  JOIN

    calendar

  ON

    b.date <= calendar.date

    AND calendar.date < b.next_date

  ORDER BY

    calendar.date)

SELECT

  DATE,

  ccc AS cumulative_contract_creation

FROM

  c

ORDER BY

  date desc

"""



contract_growth = bq_assistant.query_to_pandas_safe(query_contract_growth, max_gb_scanned=500)



contract_growth.plot(x='DATE', y='cumulative_contract_creation', kind='line', 

        figsize=(10, 8), legend=False, style='c-')

plt.title("Cumulative user-created contract deployment \n from 30.70.2015 until today", y=1.01, fontsize=20)

plt.axvline(pd.Timestamp('2018-09-27'),color='black',linestyle='--', label='After the crypto bubble "explosion"')

plt.axvline(pd.Timestamp('2016-06-18'),color='green',linestyle='--', label='DAO Hack')

plt.ylabel("count of user-created contracts (in million)", labelpad=15)

plt.xlabel("time", labelpad=15)

plt.legend()
# 2.1 Query for heavy users (preliminary analysis of users with 200 to 2000 updates)

min1 = 200

max1 = 2000

min2 = 100

max2 = 1000

min3 = 400

max3 = 4000

min4 = 50

max4 = 4000

query_accounts_updates = """

SELECT 

    address, 

    SUM(n_updates) AS updates,

    first_txn,

    last_txn,

    DATE_DIFF(DATE(last_txn),DATE(first_txn),day) AS days_active,

    (CASE WHEN address IN (SELECT DISTINCT address FROM `bigquery-public-data.crypto_ethereum.contracts`) THEN TRUE

    ELSE FALSE END) AS IsContract

FROM 

(

    SELECT 

        address, 

        count(*) AS n_updates, 

        MIN(block_timestamp) AS first_txn,

        MAX(block_timestamp) AS last_txn

        FROM

        (

            SELECT 

                DISTINCT from_address AS address, 

                block_number AS block_number, 

                block_timestamp AS block_timestamp

                FROM 

                    `bigquery-public-data.ethereum_blockchain.transactions`

                    WHERE 

                    block_number < 6400000 

                    AND 

                    block_number > 5100000

        )

    GROUP BY address

    UNION ALL

    SELECT 

        address AS address, 

        count(*) AS n_updates,

        MIN(block_timestamp) AS first_txn,

        MAX(block_timestamp) AS last_txn

        FROM 

        (

            SELECT 

                DISTINCT to_address AS address, 

                block_number AS block_number, 

                block_timestamp AS block_timestamp

                FROM 

                    `bigquery-public-data.ethereum_blockchain.transactions`

                    WHERE 

                    block_number < 6400000 

                    AND 

                    block_number > 5100000

        )

        GROUP BY address

)

WHERE n_updates >= %s 

AND

n_updates <= %s

AND

address IS NOT NULL

GROUP BY 

    address,

    first_txn,

    last_txn,

    IsContract

ORDER BY 

    updates DESC

"""



accounts_updates1 = bq_assistant.query_to_pandas_safe(query_accounts_updates % (min1, max1), max_gb_scanned=65)

heavy_users1 = accounts_updates1.loc[accounts_updates1['IsContract'] == False]

heavy_users1.describe()
# 2.1 continued: Histrogram to show the distribution of heavy users by updates (200 to 2000 updates)

import matplotlib.pyplot as plt



accounts_updates2 = bq_assistant.query_to_pandas_safe(query_accounts_updates % (min2, max2), max_gb_scanned=65)

heavy_users2 = accounts_updates2.loc[accounts_updates2['IsContract'] == False]

accounts_updates3 = bq_assistant.query_to_pandas_safe(query_accounts_updates % (min3, max3), max_gb_scanned=65)

heavy_users3 = accounts_updates3.loc[accounts_updates3['IsContract'] == False]

accounts_updates4 = bq_assistant.query_to_pandas_safe(query_accounts_updates % (min4, max4), max_gb_scanned=65)

heavy_users4 = accounts_updates4.loc[accounts_updates4['IsContract'] == False]



plt.gcf().set_size_inches(10.5, 10.5)



#subplot with 200 to 2000 updates

plt.subplot(221)

plt.plot(heavy_users1.updates)

plt.title('200 to 2000 updates')



#subplot with 100 to 1000 updates

plt.subplot(222)

plt.plot(heavy_users2.updates)

plt.title('100 to 1000 updates')



#subplot with 400 to 4000 updates

plt.subplot(223)

plt.plot(heavy_users3.updates)

plt.title('400 to 4000 updates')



#subplot with 50 to 4000 updates

plt.subplot(224)

plt.plot(heavy_users4.updates)

plt.title('50 to 4000 updates')



plt.show()
# 2.2: Bargraph of users vs contracts by udpates (200 to 2000 updates)

import matplotlib.pyplot as plt

import numpy as np



plt.gcf().set_size_inches(9,9)



#bar position

r = [0,1,2,3,4,5]



#bar labels

labels = ['200-499','500-799','800-1099','1100-1399','1400-1699','1700-2000']



#split data by IsContract

users = accounts_updates1.loc[accounts_updates1['IsContract'] == False]

contracts = accounts_updates1.loc[accounts_updates1['IsContract'] == True]



#group splitted data by update counts

users_by = users.groupby('updates').size().reset_index(name='counts')

contracts_by = contracts.groupby('updates').size().reset_index(name='counts')



#split data into selected columns

users_bars = [users_by.loc[users_by['updates']< 500]['counts'].sum(),users_by.loc[(users_by['updates']>= 500) & (users_by['updates'] < 800)]['counts'].sum(),users_by.loc[(users_by['updates']>= 800) & (users_by['updates'] < 1100)]['counts'].sum(),users_by.loc[(users_by['updates']>= 1100) & (users_by['updates'] < 1400)]['counts'].sum(),users_by.loc[(users_by['updates']>= 1400) & (users_by['updates'] < 1700)]['counts'].sum(), users_by.loc[users_by['updates'] >= 1700]['counts'].sum()]

contracts_bars = [contracts_by.loc[contracts_by['updates']< 500]['counts'].sum(),users_by.loc[(users_by['updates']>= 500) & (users_by['updates'] < 800)]['counts'].sum(),contracts_by.loc[(contracts_by['updates']>= 800) & (contracts_by['updates'] < 1100)]['counts'].sum(),users_by.loc[(users_by['updates']>= 1100) & (users_by['updates'] < 1400)]['counts'].sum(),users_by.loc[(users_by['updates']>= 1400) & (users_by['updates'] < 1700)]['counts'].sum(), contracts_by.loc[contracts_by['updates'] >= 1700]['counts'].sum()]



#plot stacked bargraph

plt.bar(r, users_bars, color='#7f6d5f', edgecolor='white', width=1, label = 'users')

plt.bar(r, contracts_bars, bottom=users_bars, color='#557f2d', edgecolor='white', width=1,label = 'contracts')

plt.xticks(r, labels, fontweight='bold')

plt.title('Stacked barplot showing users and contracts by updates (200 to 2000 updates)')

plt.ylabel("Number of users", labelpad=15)

plt.xlabel("Update range", labelpad=15)

plt.legend()

plt.show()
# 3.1 All Investors ordered by tokens (extensive query explanation below)



query_token_investors = """

WITH

  #date of first transfer per token (used in every transfer query to select transfers within x days from first transfer --> ICO date)

  first_trf AS(

  SELECT

    DISTINCT tr.token_address AS token,

    MIN(tr.block_timestamp) AS time

  FROM

    `bigquery-public-data.ethereum_blockchain.token_transfers` AS tr

  WHERE

    tr.block_number BETWEEN 510000

    AND 6400000

  GROUP BY

    tr.token_address)

  ,

    #issuer per token (issuer: most occuring from_address per token within 15 days of first token transfer)

  issuer AS (

    #select from_address with highest occurrence (occurrence = max_count) per token --> issuer

  SELECT

    DISTINCT tt.token_address AS token1,

    tt.from_address AS issuer1,

    COUNT(tt.from_address) AS c1

  FROM

    `bigquery-public-data.ethereum_blockchain.token_transfers` AS tt

  INNER JOIN (

      #select max_count per token

    SELECT

      token_address,

      MAX(c) AS maxCount

    FROM (

        #count occurrence of from_addresses (potential issuer) per token

      SELECT

        token_address,

        from_address,

        COUNT(from_address) AS c

      FROM

        `bigquery-public-data.ethereum_blockchain.token_transfers`

      JOIN

        first_trf

      ON

        token_address = first_trf.token

      WHERE

        CAST(block_timestamp AS date) <= DATE_ADD(CAST(first_trf.time AS date),INTERVAL 15 DAY)

        AND block_number BETWEEN 5100000

        AND 6400000

      GROUP BY

        token_address,

        from_address)AS tt1

    GROUP BY

      token_address) AS groupedtt

  ON

    tt.token_address = groupedtt.token_address

  JOIN

    first_trf

  ON

    tt.token_address = first_trf.token

  WHERE

    CAST(tt.block_timestamp AS date) <= DATE_ADD(CAST(first_trf.time AS date),INTERVAL 15 DAY)

    AND tt.block_number BETWEEN 5100000

    AND 6400000

  GROUP BY

    tt.token_address,

    tt.from_address,

    groupedtt.maxCount

  HAVING

    groupedtt.maxCount = c1

    AND c1 > 1)

  #select all investors who have once sent a transaction to that contract (cross-check)

SELECT

  investor,

  token_contract,

  token_issuer,

  first_transfer

FROM (

    #select all investors (investors =^ to_addresses that received a token transfer from the respective issuer)

  SELECT

    DISTINCT t.to_address AS investor,

    symbol,

    issuer.token1 AS token_contract,

    issuer.issuer1 AS token_issuer,

    first_trf.time AS first_transfer

  FROM

    `bigquery-public-data.ethereum_blockchain.token_transfers` AS t

  JOIN

    issuer

  ON

    issuer.issuer1 = t.from_address

    AND issuer.token1 = t.token_address

  JOIN

    `bigquery-public-data.ethereum_blockchain.tokens`

  ON

    address = t.token_address

  JOIN

    first_trf

  ON

    first_trf.token = issuer.token1

  WHERE

    CAST(t.block_timestamp AS date) <= DATE_ADD(CAST(first_trf.time AS date),INTERVAL 15 DAY)

    AND t.block_number BETWEEN 5100000

    AND 6400000

  GROUP BY

    investor,

    token_issuer,

    token1,

    symbol,

    first_trf.time)

  #crosscheck that investor has once made txn to token_contract

INNER JOIN

  `bigquery-public-data.ethereum_blockchain.transactions`

ON

  investor = from_address

WHERE

  to_address = token_contract

GROUP BY

  token_contract,

  investor,

  token_issuer,

  first_transfer

ORDER BY

  token_contract ASC

  """



token_investors = bq_assistant.query_to_pandas_safe(query_token_investors, max_gb_scanned=500)

tok_inv = token_investors.head(5)

tok_inv.append(token_investors.tail(5))
# 3.2 Unique investors who have invested in at least 5 different tokens

import matplotlib.pyplot as plt

investments_per_investor = token_investors.groupby('investor').size().reset_index(name='investment_count')

inv_five = investments_per_investor.loc[investments_per_investor['investment_count'] >= 5].groupby('investment_count').size().reset_index(name = 'investors')

inv_five.describe()
# 3.2 continued: Bargraph of investments per investor

import matplotlib.pyplot as plt



plt.gcf().set_size_inches(9,9)



#subplot with investment_count >= 5

plt.subplot(211)

plt.bar(inv_five.investment_count,inv_five.investors)

plt.title('Investors per Investment count (count >= 5)', y=1.01, fontsize=15)

plt.ylabel('Investors per investment_count',labelpad=15, fontsize=12)



#subplot with investment_count >= 35

plt.subplot(212)

plt.bar(inv_five.loc[inv_five['investment_count'] >= 35].investment_count, inv_five.loc[inv_five['investment_count'] >= 35].investors)

plt.title('Investors per Investment count (count >= 35)', y=1.01, fontsize=15)

plt.xlabel('Investment_count', fontsize=12)

plt.ylabel('Investors per investment_count',labelpad=15, fontsize=12)

plt.show()
# 3.3 Token investors by invesment count (>=5) and by eth.invested

query_eth_by_investor = """

WITH

  first_trf AS(

  SELECT

    DISTINCT tr.token_address AS token,

    MIN(tr.block_timestamp) AS time

  FROM

    `bigquery-public-data.ethereum_blockchain.token_transfers` AS tr

  WHERE

    tr.block_number BETWEEN 510000

    AND 6400000

  GROUP BY

    tr.token_address)

  #issuer per token (issuer =^ most occuring from_address per token within 15 days of first token transfer)

  ,

  issuer AS (

    #select from_address with highest occurrence (occurrence = max_count) per token --> issuer

  SELECT

    DISTINCT tt.token_address AS token1,

    tt.from_address AS issuer1,

    COUNT(tt.from_address) AS c1

  FROM

    `bigquery-public-data.ethereum_blockchain.token_transfers` AS tt

  INNER JOIN (

      #select max_count per token

    SELECT

      token_address,

      MAX(c) AS maxCount

    FROM (

        #count occurrence of from_addresses (potential issuer) per token

      SELECT

        token_address,

        from_address,

        COUNT(from_address) AS c

      FROM

        `bigquery-public-data.ethereum_blockchain.token_transfers`

      JOIN

        first_trf

      ON

        token_address = first_trf.token

      WHERE

        CAST(block_timestamp AS date) <= DATE_ADD(CAST(first_trf.time AS date),INTERVAL 15 DAY)

        AND block_number BETWEEN 5100000

        AND 6400000

      GROUP BY

        token_address,

        from_address)AS tt1

    GROUP BY

      token_address) AS groupedtt

  ON

    tt.token_address = groupedtt.token_address

  JOIN

    first_trf

  ON

    tt.token_address = first_trf.token

  WHERE

    CAST(tt.block_timestamp AS date) <= DATE_ADD(CAST(first_trf.time AS date),INTERVAL 15 DAY)

    AND tt.block_number BETWEEN 5100000

    AND 6400000

  GROUP BY

    tt.token_address,

    tt.from_address,

    groupedtt.maxCount

  HAVING

    groupedtt.maxCount = c1

    AND c1 > 1)

  #select all investors who have once sent a transaction to that contract (cross-check)

SELECT

  DISTINCT investor,

  COUNT(token_contract) as investments,

  sum(value)/power(10,18) as eth_invested

FROM

(SELECT

  investor,

  token_contract,

  token_issuer,

  first_transfer,

  value

FROM (

    #select all investors (investors =^ to_addresses that received a token transfer from the respective issuer)

  SELECT

    DISTINCT t.to_address AS investor,

    symbol,

    issuer.token1 AS token_contract,

    issuer.issuer1 AS token_issuer,

    first_trf.time AS first_transfer

  FROM

    `bigquery-public-data.ethereum_blockchain.token_transfers` AS t

  JOIN

    issuer

  ON

    issuer.issuer1 = t.from_address

    AND issuer.token1 = t.token_address

  JOIN

    `bigquery-public-data.ethereum_blockchain.tokens`

  ON

    address = t.token_address

  JOIN

    first_trf

  ON

    first_trf.token = issuer.token1

  WHERE

    CAST(t.block_timestamp AS date) <= DATE_ADD(CAST(first_trf.time AS date),INTERVAL 15 DAY)

    AND t.block_number BETWEEN 5100000

    AND 6400000

  GROUP BY

    investor,

    token_issuer,

    token1,

    symbol,

    first_trf.time)

  #crosscheck that investor has once made txn to token_contract

INNER JOIN

  `bigquery-public-data.ethereum_blockchain.transactions`

ON

  investor = from_address

WHERE

  to_address = token_contract

GROUP BY

  token_contract,

  investor,

  token_issuer,

  first_transfer,

  value)

GROUP BY

  investor

HAVING

  investments >= 2

ORDER BY

  eth_invested DESC

  """

eth_by_investor = bq_assistant.query_to_pandas(query_eth_by_investor)

eth_by_investor.head()
# 3.3 continued: Plots of token investors by investment value

import matplotlib.pyplot as plt



plt.gcf().set_size_inches(9,9)



#round all investments to nearest float with one decimal place and group data by count of eth invested

inv_value_grouped = eth_by_investor.round(1).groupby('eth_invested').size().reset_index(name = 'count')



#subplot including all ether values invested

plt.subplot(211)

plt.plot(inv_value_grouped.eth_invested)

plt.title('Investors per Investment Value', y=1.01, fontsize=15)

plt.ylabel('Investment value (per unique investor)',labelpad=15, fontsize=12)



#limit dataset to ether value from 1 to 100

inv_value_lim = inv_value_grouped.loc[(inv_value_grouped['eth_invested']>= 1) & (inv_value_grouped['eth_invested'] < 100)]



#subplot including ether values invested from 1 to 100

plt.subplot(212)

plt.plot(inv_value_lim.eth_invested)

plt.title('Investors per Investment Value (1 to 100)', y=1.01, fontsize=15)

plt.xlabel('Investors', fontsize=12)

plt.ylabel('Investment value (per unique investor)',labelpad=15, fontsize=12)



plt.show()