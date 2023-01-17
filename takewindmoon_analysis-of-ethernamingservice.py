from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data","crypto_ethereum") #NEW NAME, old name in examples doesn't work

# not actually useful! also data not in Kaggle

                              # bigquery-public-data","crypto_ethereum_classic") #bigquery-public-data.crypto_ethereum_classic.logs

# "bigquery-public-data", "github_repos")

#'', project='bigquery-public-data'



# What else people paid after they paid to the ENS contract.



FROM_BLOCK_TIMESTAMP='2019-01-01 05:41:19'

TO_BLOCK_TIMESTAMP='2019-03-01 05:41:19'





QUERY="""

SELECT trans_down.*

	,RANK() OVER (

		PARTITION BY trans.from_address

		,trans.block_timestamp ORDER BY trans_down.block_timestamp

		) AS rank

	,datetime_diff(cast(trans_down.block_timestamp AS DATETIME), cast(trans.block_timestamp AS DATETIME), hour) AS HoursGap

FROM `bigquery - PUBLIC - data.crypto_ethereum.transactions` AS trans

INNER JOIN `bigquery - PUBLIC - data.crypto_ethereum.transactions` AS trans_down ON trans.from_address = trans_down.from_address

	AND trans_down.to_address NOT IN ({contract_list})

	AND (trans_down.block_timestamp > trans.block_timestamp)

WHERE (

		trans.block_timestamp >= {from_timestamp} 

		

		)

AND trans.to_address IN ({contract_list})

""".format(from_timestamp=FROM_BLOCK_TIMESTAMP, to_timestamp=TO_BLOCK_TIMESTAMP, 

           contract_list="'0x314159265dd8dbb310642f98f50c066173c1259b'")#",'0xA62142888ABa8370742bE823c1782D17A0389Da1','0xa62142888aba8370742be823c1782d17a0389da1','0xf056F435Ba0CC4fCD2F1B17e3766549fFc404B94'")

bq_assistant.estimate_query_size(QUERY) #estimate_gigabytes_scanned("SELECT Id FROM `bigquery-public-data.hacker_news.stories`", client)



transactions = bq_assistant.query_to_pandas_safe(QUERY)

transactions



# What they did before paying to the ENS contract

QUERY="""

SELECT trans_up.*

	,trans_down.to_address AS Destination_address

FROM `bigquery - PUBLIC - data.crypto_ethereum.transactions` AS trans

INNER JOIN `bigquery - PUBLIC - data.crypto_ethereum.transactions` AS trans_down ON trans.from_address = trans_down.from_address

	AND (

		trans_down.block_timestamp >= '2019-01-01 05:41:19'

		AND trans_down.block_timestamp <= '2019-04-08 05:41:19'

		)

INNER JOIN `bigquery - PUBLIC - data.crypto_ethereum.transactions` AS trans_up ON trans.from_address = trans_up.to_address

	AND (

		trans_up.block_timestamp >= '2018-12-01 05:41:19'

		AND trans_up.block_timestamp <= '2019-03-08 05:41:19'

		)

WHERE (

		trans.block_timestamp >= {from_timestamp} 

  AND trans.block_timestamp <= {to_timestamp}

		

		)

AND trans.to_address IN ({contract_list})

 

  AND from_address IN ({contract_list})

""".format(from_timestamp=FROM_BLOCK_TIMESTAMP, to_timestamp=TO_BLOCK_TIMESTAMP, 

           contract_list="'0x314159265dd8dbb310642f98f50c066173c1259b'")#",'0xA62142888ABa8370742bE823c1782D17A0389Da1','0xa62142888aba8370742be823c1782d17a0389da1','0xf056F435Ba0CC4fCD2F1B17e3766549fFc404B94'")

bq_assistant.estimate_query_size(QUERY) #estimate_gigabytes_scanned("SELECT Id FROM `bigquery-public-data.hacker_news.stories`", client)



transactions = bq_assistant.query_to_pandas_safe(QUERY)

transactions