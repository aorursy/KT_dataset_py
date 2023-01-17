import bq_helper
# Define the tokens to analyze

tokens = {

    'OMG': '0xd26114cd6EE289AccF82350c8d8487fedB8A0C07'.lower(),

    'BAT': '0x0d8775f648430679a709e98d2b0cb6250d2887ef'.lower(),

    'BNB': '0xB8c77482e45F1F44dE1745F52C74426C631bDD52'.lower(),

    'ZIL': '0x05f4a42e251f2d52b8ed15e9fedaacfcef1fad27'.lower(),

    'DAI': '0x89d24a6b4ccb1b6faa2625fe562bdd9a23260359'.lower()

}
# Helper object for BigQuery Ethereum dataset

eth = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                               dataset_name="crypto_ethereum",

                               max_wait_seconds=1800)
query = """

-- Compute total tokens received for each address

WITH receivedTx AS (

    SELECT tx.to_address AS addr, 

           tx.token_address AS caddr, 

           tx.block_number AS block,

           SUM(SAFE_CAST(tx.value AS FLOAT64)/POWER(10,18)) AS amount_received

    FROM `bigquery-public-data.crypto_ethereum.token_transfers` AS tx

    WHERE lower(tx.token_address) = '{t}'

        -- Exclude implicit token burns

        AND tx.to_address != '0x0000000000000000000000000000000000000000'

    GROUP BY 1, 2, 3

),



-- Compute total tokens sent for each address

sentTx AS (

    SELECT tx.from_address AS addr, 

           tx.token_address AS caddr, 

           tx.block_number AS block,

           SUM(SAFE_CAST(tx.value AS FLOAT64)/POWER(10,18)) AS amount_sent

    FROM `bigquery-public-data.crypto_ethereum.token_transfers` AS tx

    WHERE lower(tx.token_address) = '{t}'

    GROUP BY 1, 2, 3

),



-- Compute each block's flows

blockBalances AS (

    SELECT r.block,

           SUM(r.amount_received) AS received,

           SUM(s.amount_sent) AS sent

    FROM receivedTx AS r, sentTx AS s

    GROUP BY r.block

)



-- Compute a rolling sum across all blocks

SELECT bb.block, 

       bd.timestamp,

       --token_address,

       sum(bb.received) 

           OVER (ORDER BY block ROWS UNBOUNDED PRECEDING) AS total_received,

       sum(bb.sent) 

           OVER (ORDER BY block ROWS UNBOUNDED PRECEDING) AS total_sent

FROM blockBalances bb,

     `bigquery-public-data.crypto_ethereum.blocks` bd

WHERE bb.block = bd.number

ORDER BY block ASC

"""



# Estimate how big this query will be

eth.estimate_query_size(query.format(t=tokens['BNB']))
%%time



# Store the results into a dict { token: DataFrame }

dfs = { t: eth.query_to_pandas_safe(query.format(t=tokens.get(t)), max_gb_scanned=35)

       for t in tokens.keys() }
dfs['BNB'].head()
import matplotlib.pyplot as plt



def plotFor(token, dfs):

    plt.figure(figsize=(16, 9))



    df = dfs[token].copy()

    df['net'] = df['total_received'] - df['total_sent']

    

    plt.plot('timestamp', 'total_sent', data=df)

    plt.plot('timestamp', 'total_received', data=df)

    plt.plot('timestamp', 'net', data=df)

    plt.title("Historical cummulative transactions for {}".format(token))

    plt.legend()

    

    numBlocksBelowZero = sum(df['net'] < 0)

    

    print("Blocks with negantive net: ", numBlocksBelowZero)

    

plotFor('BNB', dfs)
plotFor('DAI', dfs)
plotFor('OMG', dfs)
plotFor('BAT', dfs)
plotFor('ZIL', dfs)