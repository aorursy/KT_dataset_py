# Block range in the original article: 4605167 to 5374870
# Filtering by block_timestamp is more cost efficient than by block_number because the tables are partitioned by block_timestamp
# block_number = 4605167 -> block_timestamp == "2017-11-23 05:41:19" 
# block_number = 5374870 -> block_timestamp == "2018-04-03 19:53:46" 

FROM_BLOCK = 4605167
TO_BLOCK = 5374870
FROM_BLOCK_TIMESTAMP = "'2018-01-01 05:41:19'" # "'2017-11-23 05:41:19'"
TO_BLOCK_TIMESTAMP = "'2018-04-03 19:53:46'"
MY_TIMEOUT = 300

# relevant Events signatures
events_signatures = {
    'AuctionCreated' : 'AuctionCreated(uint256,uint256,uint256,uint256)', # AuctionCreated(uint256 tokenId, uint256 startingPrice, uint256 endingPrice, uint256 duration);
    'AuctionSuccessful' : 'AuctionSuccessful(uint256,uint256,address)', # AuctionSuccessful(uint256 tokenId, uint256 totalPrice, address winner);
    'AuctionCancelled' : 'AuctionCancelled(uint256)', # AuctionCancelled(uint256 tokenId);
    'Pause' : 'Pause()',
    'Unpause' : 'Unpause()',
    'Transfer' : 'Transfer(address,address,uint256)', # Transfer(address from, address to, uint256 tokenId);
    'Approval' : 'Approval(address,address,uint256)', # Approval(address owner, address approved, uint256 tokenId);
    'ContractUpgrade' : 'ContractUpgrade(address)',
    'Birth' : 'Birth(address,uint256,uint256,uint256,uint256)', # Birth(address owner, uint256 kittyId, uint256 matronId, uint256 sireId, uint256 genes);
    'Pregnant' : 'Pregnant(address,uint256,uint256,uint256)' # Pregnant(address owner, uint256 matronId, uint256 sireId, uint256 cooldownEndBlock);
}
events_hashes = {'0a5311bd2a6608f08a180df2ee7c5946819a649b204b554bb8e39825b2c50ad5': 'Birth',
 '241ea03ca20251805084d27d4440371c34a0b85ff108f6bb5611248f73818b80': 'Pregnant',
 '2809c7e17bf978fbc7194c0a694b638c4215e9140cacc6c38ca36010b45697df': 'AuctionCancelled',
 '450db8da6efbe9c22f2347f7c2021231df1fc58d3ae9a2fa75d39fa446199305': 'ContractUpgrade',
 '4fcc30d90a842164dd58501ab874a101a3749c3d4747139cefe7c876f4ccebd2': 'AuctionSuccessful',
 '6985a02210a168e66602d3235cb6db0e70f92b3ba4d376a33c0f3d9434bff625': 'Pause',
 '7805862f689e2f13df9f062ff482ad3ad112aca9e0847911ed832e158c525b33': 'Unpause',
 '8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925': 'Approval',
 'a9c8dfcda5664a5a124c713e386da27de87432d5b668e79458501eb296389ba7': 'AuctionCreated',
 'ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef': 'Transfer'}
coreContract = '0x06012c8cf97bead5deae237070f9587f8e7a266d'
contracts = {
    coreContract : 'core',
    '0xc7af99fe5513eb6710e6d5f44f9989da40f27f26' : 'siringAuction',
    '0xb1690c08e213a35ed9bab7b318de14420fb57d8c' : 'saleAuction',
}
from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('max_colwidth', 70)
#events = pd.read_pickle('ck-data/events_4605167-to-5374870.pickle.gz')
client = bigquery.Client()
query = """
SELECT 
  transaction_hash AS transactionHash,
  address,
  data,
  topics,
  block_timestamp,
  block_number AS blockNumber_dec
FROM
  `bigquery-public-data.ethereum_blockchain.logs` AS events
WHERE TRUE
  AND block_timestamp >= {from_block_ts} 
  AND block_timestamp <= {to_block_ts}
  AND address IN ({contract_list})
""".format(from_block_ts=FROM_BLOCK_TIMESTAMP, to_block_ts=TO_BLOCK_TIMESTAMP, contract_list="'0x06012c8cf97bead5deae237070f9587f8e7a266d','0xc7af99fe5513eb6710e6d5f44f9989da40f27f26','0xb1690c08e213a35ed9bab7b318de14420fb57d8c'")
print(query)
query_job = client.query(query)
iterator = query_job.result(timeout=MY_TIMEOUT)
rows = list(iterator)
events = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
events['contract'] = events['address'].apply(lambda x: contracts[x])
events['event'] = events['topics'].apply(lambda x: events_hashes[x[0][2:]])
events.head(10)
print('Block range: ' + str(events.blockNumber_dec.min()) + ' to ' + str(events.blockNumber_dec.max()))
events.groupby(['contract','event']).transactionHash.count()
event_counts = events.groupby(['contract','event']).transactionHash.count()
event_counts.sort_values().plot(kind='barh', figsize=(8, 6))
event_counts_df = event_counts.reset_index()
event_counts_df.columns = ['contract', 'event', 'count']
event_counts_df
transfer_count = event_counts_df[event_counts_df['event']=='Transfer'].iloc[0]['count']

sale_auction_cancelled_count = event_counts_df[(event_counts_df['event']=='AuctionCancelled') & \
                                             (event_counts_df['contract']=='saleAuction')].iloc[0]['count']

siring_auction_cancelled_count = event_counts_df[(event_counts_df['event']=='AuctionCancelled') & \
                                             (event_counts_df['contract']=='siringAuction')].iloc[0]['count']

sale_auction_created_count = event_counts_df[(event_counts_df['event']=='AuctionCreated') & \
                                             (event_counts_df['contract']=='saleAuction')].iloc[0]['count']

siring_auction_created_count = event_counts_df[(event_counts_df['event']=='AuctionCreated') & \
                                             (event_counts_df['contract']=='siringAuction')].iloc[0]['count']

sale_auction_successful_count = event_counts_df[(event_counts_df['event']=='AuctionSuccessful') & \
                                             (event_counts_df['contract']=='saleAuction')].iloc[0]['count']

siring_auction_successful_count = event_counts_df[(event_counts_df['event']=='AuctionSuccessful') & \
                                             (event_counts_df['contract']=='siringAuction')].iloc[0]['count']

birth_count = event_counts_df[(event_counts_df['event']=='Birth')].iloc[0]['count']

pregnant_count = event_counts_df[(event_counts_df['event']=='Pregnant')].iloc[0]['count']
(transfer_count - \
sale_auction_cancelled_count - \
siring_auction_cancelled_count - \
sale_auction_created_count - \
siring_auction_created_count - \
sale_auction_successful_count - \
birth_count ) / \
transfer_count
pregnant_count/siring_auction_successful_count
events['contract-event'] = events['contract'] + events['event']
events['block-group'] = events['blockNumber_dec'].apply(lambda x: int(x/1000))
areaplot = events.groupby(['block-group','contract-event']).transactionHash.count().reset_index().pivot(index='block-group', columns='contract-event', values='transactionHash')#.plot.area()
areaplot.plot.area()
plt.legend(loc=1)
areaplot.divide(areaplot.sum(axis=1), axis=0).plot.area(figsize=(16, 9))
plt.legend(loc=1)
transaction_birthcount = (events[events['event']=='Birth'])[['transactionHash','event']].groupby(['transactionHash']).count().reset_index()
transaction_birthcount[transaction_birthcount['event']>1].head()
#births = pd.read_pickle('ck-data/births_4605167-to-5374870.pickle.gz')
client = bigquery.Client()
query = """
SELECT 
  events.transaction_hash AS transactionHash,
  events.data,
  events.block_timestamp,
  events.block_number AS blockNumber_dec,
  txns.from_address AS midwife, 
  txns.to_address AS midwife_smartcontract, 
  txns.gas_price AS gasPrice, 
  txns.receipt_gas_used AS gasUsed
FROM
  `bigquery-public-data.ethereum_blockchain.logs` AS events
INNER JOIN
  `bigquery-public-data.ethereum_blockchain.transactions` AS txns
ON
  events.transaction_hash = txns.hash
WHERE TRUE
  AND events.block_timestamp >= {from_block_ts} 
  AND events.block_timestamp <= {to_block_ts}
  AND txns.block_timestamp >= {from_block_ts} --might seem redundant, but because of partitioning this reduces cost
  AND txns.block_timestamp <= {to_block_ts} --might seem redundant, but because of partitioning this reduces cost
  AND events.address = '0x06012c8cf97bead5deae237070f9587f8e7a266d'
  AND events.topics[OFFSET(0)] = '0x0a5311bd2a6608f08a180df2ee7c5946819a649b204b554bb8e39825b2c50ad5'
""".format(from_block_ts=FROM_BLOCK_TIMESTAMP, 
           to_block_ts=TO_BLOCK_TIMESTAMP)
print(query)
query_job = client.query(query)
iterator = query_job.result(timeout=MY_TIMEOUT)
rows = list(iterator)
births = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
births.head(10)
births['owner'] = '0x' + births['data'].apply(lambda x: x[26:66])
births['kittyId'] = births['data'].apply(lambda x: x[66:130])
births['kittyId_dec'] = births['kittyId'].apply(lambda x: int(x,16))
births['matronId'] = births['data'].apply(lambda x: x[130:194])
births['matronId_dec'] = births['matronId'].apply(lambda x: int(x,16))
births['sireId'] = births['data'].apply(lambda x: x[194:258])
births['sireId_dec'] = births['sireId'].apply(lambda x: int(x,16))
births['kittyGenes'] = births['data'].apply(lambda x: x[258:322])
births['block-group'] = births['blockNumber_dec'].apply(lambda x: int(x/1000))
maxBirths = births.groupby(['transactionHash']).transactionHash.count().max()
births.groupby(['transactionHash']).transactionHash.count().hist(bins=range(maxBirths+2))
#this was in the original notebook; no longer needed because we now query this field on SELECT
#births['midwife'] = births['transaction'].apply(lambda x: eval(x)['from'])
#this was in the original notebook; no longer needed because we now query this field on SELECT
#births['midwife_smartcontract'] = births['transaction'].apply(lambda x: eval(x)['to'])
#this was in the original notebook; no longer needed because we now query this field on SELECT
#births['gasPrice'] = births['transaction'].apply(lambda x: int(eval(x)['gasPrice'],16))
births['fee'] = births['gasUsed'] * births['gasPrice'] * 1E-18
AxiomZenAccounts = ['0xa21037849678af57f9865c6b9887f4e339f6377a','0xba52c75764d6f594735dc735be7f1830cdf58ddf']
allTimeTopMidwives = births.groupby(['midwife']).data.count().\
                sort_values(ascending=False)
len(allTimeTopMidwives)
allTimeTopMidwives = set(allTimeTopMidwives.head(10).index.values)
allTimeTopMidwives
births['midwife-group'] = births['midwife'].apply(lambda x: '1- AxiomZen' \
                                                if x in AxiomZenAccounts \
                                                else '2- All Time Top 10' if x in allTimeTopMidwives \
                                                else '3- Other')
areaplot = births.groupby(['block-group','midwife-group']).transactionHash.count().reset_index().pivot(index='block-group', columns='midwife-group', values='transactionHash')
areaplot.divide(areaplot.sum(axis=1), axis=0).plot.area(figsize=(16, 9))
plt.legend(loc=1)
areaplot.plot.area(figsize=(16, 9))
plt.legend(loc=1)
movingTopFiveMidwives = births.groupby(['block-group','midwife']).data.count().reset_index().\
                sort_values(by=['block-group','data'],ascending=False).groupby(['block-group']).head(5)
movingTopFiveMidwives = set(movingTopFiveMidwives.midwife.values)
len(movingTopFiveMidwives)
births['midwife-group'] = births['midwife'].apply(lambda x: '1- AxiomZen' \
                                                if x in AxiomZenAccounts \
                                                else '2- Moving Top 5' if x in movingTopFiveMidwives \
                                                else '3- Other')
areaplot = births.groupby(['block-group','midwife-group']).transactionHash.count().reset_index().pivot(index='block-group', columns='midwife-group', values='transactionHash')
areaplot.divide(areaplot.sum(axis=1), axis=0).plot.area(figsize=(16, 9))
plt.legend(loc=1)
births['type-of-call'] = births['midwife_smartcontract'].apply(lambda x: 'Direct call' \
                                                if x == '0x06012c8cf97bead5deae237070f9587f8e7a266d' \
                                                else 'Intermediary smart-contract')
areaplot = births.groupby(['block-group','type-of-call']).transactionHash.count().reset_index().pivot(index='block-group', columns='type-of-call', values='transactionHash')
areaplot.divide(areaplot.sum(axis=1), axis=0).plot.area(figsize=(16, 9))
plt.legend(loc=1)
areaplot.plot.area(figsize=(16, 9))
plt.legend(loc=1)
births[births['midwife'].isin(allTimeTopMidwives)].groupby(['midwife']).data.count().sort_values().plot(kind='barh')
plt.xlabel('kitties delivered')
#4832686 first block of the year 2018
births_2018 = births[births['blockNumber_dec']>=4832686]
TopMidwives_2018 = births_2018.groupby(['midwife']).data.count().\
                sort_values(ascending=False).head(10)
TopMidwives_2018 = set(TopMidwives_2018.index.values)
TopMidwives_2018
births_2018[births_2018['midwife'].isin(TopMidwives_2018)].groupby(['midwife']).data.count().sort_values().plot(kind='barh')
plt.xlabel('Number of kitties delivered')
count = pd.DataFrame(births_2018.groupby(['transactionHash']).transactionHash.count())
fees = births_2018.groupby(['transactionHash']).fee.max()
midwife = births_2018.groupby(['transactionHash']).midwife.max()
midwife_smartcontract = births_2018.groupby(['transactionHash']).midwife_smartcontract.max()
gasUsed = births_2018.groupby(['transactionHash']).gasUsed.max()
gasPrice = births_2018.groupby(['transactionHash']).gasPrice.max()
df_profitability = count.join(fees).join(midwife).join(midwife_smartcontract).join(gasUsed).join(gasPrice)
df_profitability.columns = ['kitties_delivered','fee','midwife','midwife_smartcontract','gasUsed','gasPrice']
df_profitability['revenue'] = df_profitability['kitties_delivered'] * 0.008
df_profitability['profit'] = df_profitability['revenue'] - df_profitability['fee']
df_profitability['code_efficiency'] = df_profitability['kitties_delivered']/df_profitability['gasUsed']*1e6
df_profitability['efficiency'] = df_profitability['profit']/df_profitability['kitties_delivered']
len(df_profitability)
MY_TIMEOUT=6000
#births = pd.read_pickle('ck-data/births_4605167-to-5374870.pickle.gz')
client = bigquery.Client()
query = """

SELECT DISTINCT
  failed_txns.hash AS transactionHash, 
  failed_txns.from_address AS midwife, 
  failed_txns.to_address AS midwife_smartcontract, 
  failed_txns.gas_price AS gasPrice, 
  failed_txns.receipt_gas_used AS gasUsed
FROM
  `bigquery-public-data.ethereum_blockchain.logs` AS events
INNER JOIN
  `bigquery-public-data.ethereum_blockchain.transactions` AS events_txns
ON
  events.transaction_hash = events_txns.hash
INNER JOIN
  `bigquery-public-data.ethereum_blockchain.transactions` AS failed_txns
ON
  events_txns.from_address = failed_txns.from_address
  AND events_txns.to_address = failed_txns.to_address
WHERE TRUE
  AND events.block_timestamp >= {from_block_ts} 
  AND events.block_timestamp <= {to_block_ts}
  AND events_txns.block_timestamp >= {from_block_ts} --might seem redundant, but because of partitioning this reduces cost
  AND events_txns.block_timestamp <= {to_block_ts} --might seem redundant, but because of partitioning this reduces cost
  AND failed_txns.block_timestamp >= {from_block_ts} --might seem redundant, but because of partitioning this reduces cost
  AND failed_txns.block_timestamp <= {to_block_ts} --might seem redundant, but because of partitioning this reduces cost
  AND events.address = '0x06012c8cf97bead5deae237070f9587f8e7a266d'
  AND events.topics[OFFSET(0)] = '0x0a5311bd2a6608f08a180df2ee7c5946819a649b204b554bb8e39825b2c50ad5'
  AND (
    (
    failed_txns.to_address = '0x06012c8cf97bead5deae237070f9587f8e7a266d' 
    AND 
    SUBSTR(failed_txns.input,0,10) = '0x88c2a0bf'
    )
    OR
    failed_txns.to_address <> '0x06012c8cf97bead5deae237070f9587f8e7a266d' 
  )

EXCEPT DISTINCT
(
SELECT DISTINCT
  txns.hash AS transactionHash, 
  txns.from_address AS midwife, 
  txns.to_address AS midwife_smartcontract, 
  txns.gas_price AS gasPrice, 
  txns.receipt_gas_used AS gasUsed
FROM
  `bigquery-public-data.ethereum_blockchain.logs` AS events
INNER JOIN
  `bigquery-public-data.ethereum_blockchain.transactions` AS txns
ON
  events.transaction_hash = txns.hash
WHERE TRUE
  AND events.block_timestamp >= {from_block_ts} 
  AND events.block_timestamp <= {to_block_ts}
  AND txns.block_timestamp >= {from_block_ts} --might seem redundant, but because of partitioning this reduces cost
  AND txns.block_timestamp <= {to_block_ts} --might seem redundant, but because of partitioning this reduces cost
  AND events.address = '0x06012c8cf97bead5deae237070f9587f8e7a266d'
  AND events.topics[OFFSET(0)] = '0x0a5311bd2a6608f08a180df2ee7c5946819a649b204b554bb8e39825b2c50ad5'
)

""".format(from_block_ts="'2018-01-01 00:00:00'", 
           to_block_ts=TO_BLOCK_TIMESTAMP)
print(query)
query_job = client.query(query)
iterator = query_job.result(timeout=MY_TIMEOUT)
rows = list(iterator)
df_failed_midwifing_txns = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df_failed_midwifing_txns.head(10)
# #dataset of all transactions from midwife to midwife-smartcontract
# #this includes transactions to the core game smart contract that are not calls to giveBirth, so we remove those
# df_all_midwifing_txns = pd.read_pickle('ck-data/midwives-txns_4605167-to-5374870.pickle.gz')
# #drop transactions to the core game that are not calls to giveBirth
# df_all_midwifing_txns = df_all_midwifing_txns[((df_all_midwifing_txns['to']=='0x06012c8cf97bead5deae237070f9587f8e7a266d') & \
#                                                (df_all_midwifing_txns['input'].apply(lambda x: x[:10]=='0x88c2a0bf'))) | \
#                                                (df_all_midwifing_txns['to']!='0x06012c8cf97bead5deae237070f9587f8e7a266d')]
# df_all_midwifing_txns['blockNumber_dec'] = df_all_midwifing_txns['blockNumber'].apply(lambda x: int(x,16))

#SELECT ONLY 2018
#df_all_midwifing_txns = df_all_midwifing_txns[df_all_midwifing_txns['blockNumber_dec']>=4832686]
#df_all_midwifing_txns.head(2)
#df_failed_midwifing_txns = (df_all_midwifing_txns[~df_all_midwifing_txns['hash'].isin(df_profitability.index)])[['hash','from','to','gasUsed','gasPrice']]
#df_failed_midwifing_txns.columns = ['transactionHash','midwife','midwife_smartcontract','gasUsed','gasPrice']
#df_failed_midwifing_txns = df_failed_midwifing_txns[df_failed_midwifing_txns['midwife'].isin(allTimeTopMidwives)]
#df_failed_midwifing_txns['gasPrice'] = df_failed_midwifing_txns['gasPrice'].apply(lambda x: int(x,16))
df_failed_midwifing_txns['kitties_delivered'] = 0
df_failed_midwifing_txns['revenue'] = 0
df_failed_midwifing_txns['fee'] = df_failed_midwifing_txns['gasUsed'] * df_failed_midwifing_txns['gasPrice'] * 1e-18
df_failed_midwifing_txns['profit'] = -df_failed_midwifing_txns['fee']
df_failed_midwifing_txns = df_failed_midwifing_txns.set_index('transactionHash')
len(df_failed_midwifing_txns)
df_profitability = df_profitability.append(df_failed_midwifing_txns, sort=True)
len(df_profitability)
df_profitability[df_profitability['midwife'].isin(TopMidwives_2018)].groupby(['midwife']).revenue.sum().sort_values().plot(kind='barh')
plt.xlabel('Revenue (ETH)')
df_profitability[df_profitability['midwife'].isin(TopMidwives_2018)].groupby(['midwife']).profit.sum().sort_values().plot(kind='barh')
plt.xlabel('Profit (ETH)')
efficiency_plot = df_profitability[df_profitability['midwife'].isin(TopMidwives_2018)].groupby(['midwife_smartcontract']).code_efficiency.mean().sort_values()
efficiency_plot.divide(efficiency_plot.max(),axis=0).plot(kind='barh')
plt.xlabel('Code efficiency')
efficiency_plot = df_profitability[df_profitability['midwife'].isin(TopMidwives_2018)].groupby(['midwife']).efficiency.mean().sort_values()
efficiency_plot.divide(efficiency_plot.max(),axis=0).plot(kind='barh')
plt.xlabel('Efficiency')
df_profitability[(df_profitability['midwife']=='0x05be6e1f661dacd4630e1ebe2ffce5bfb962076f') & \
                 (df_profitability['kitties_delivered']==0) & \
                 (df_profitability['midwife_smartcontract']=='0x39243a59d34169eeb0cac2752a21b982408a0194') & \
                 (df_profitability['gasUsed']>55000)].head()
df_profitability[(df_profitability['midwife']=='0x05be6e1f661dacd4630e1ebe2ffce5bfb962076f') & \
                 (df_profitability['kitties_delivered']==0) & \
                 (df_profitability['midwife_smartcontract']=='0x39243a59d34169eeb0cac2752a21b982408a0194') & \
                 (df_profitability['gasUsed']>55000)].profit.sum()
births_2018[births_2018['midwife'].isin(TopMidwives_2018)].groupby(['midwife_smartcontract']).data.count().sort_values().plot(kind='barh')
plt.xlabel('Number of kitties delivered')
TopMidwivesSmartContracts_2018 = births_2018[births_2018['midwife'].isin(TopMidwives_2018)].\
                groupby(['midwife_smartcontract']).data.count().\
                sort_values(ascending=False).head(10)
TopMidwivesSmartContracts_2018 = set(TopMidwivesSmartContracts_2018.index.values)
TopMidwivesSmartContracts_2018
import seaborn as sns
heatmap = pd.DataFrame(births_2018[(births_2018['midwife'].isin(TopMidwives_2018)) & \
                                   (births_2018['midwife_smartcontract'].isin(TopMidwivesSmartContracts_2018))]\
                       .groupby(['midwife','midwife_smartcontract']).data.count()).reset_index().pivot(index='midwife', columns='midwife_smartcontract', values='data')
heatmap = heatmap.fillna(0)
#reorder rows and columns to make clusters more evident
heatmap = heatmap[['0x06012c8cf97bead5deae237070f9587f8e7a266d',
 '0x39243a59d34169eeb0cac2752a21b982408a0194',
 '0x79b2f239bc75755a6bf38f55e697356fe7c61bec',
 '0x7449fe237fc0873481230380af32a73910ad2afd',
 '0xb81e1c30149813d743d495f68341c5129eb73e88',
 '0x903625318d13a3529fb4309fd57dd8d105fcd39c',
 '0xd18785571ae7f3b100e5b8788e3827120282f170',
 '0xc5f60fa4613493931b605b6da1e9febbdeb61e16',
 '0xa08f3503933d050ce415ca1db26ddba6a2231e3e',
 '0xdc969012dcc40402316a1964dc77ae85cbb33e2d'
]]
heatmap = heatmap.reindex([ '0xa21037849678af57f9865c6b9887f4e339f6377a',
 '0x05be6e1f661dacd4630e1ebe2ffce5bfb962076f',
 '0x74f42f97a229213343cbe0be747dfe4b705876cb',
 '0xb7f819b983e0cbb0316786d7fba12e3b1e58da5f',
 '0x6fc9bcb6091c01d6d2a530955e633b894ae48256',
 '0x80cfd274937d40c5e3d0e910a81d3330f3c10898',
 '0xc5b373618d4d01a38f822f56ca6d2ff5080cc4f2',
 '0xed9878336d5187949e4ca33359d2c47c846c9dd3',
 '0xd294209c4132b227902b03cf8c7c8d4d4a780eb4',
 '0xf3461cc074cd21b60cbf393050c4990332215186'])
sns.heatmap(heatmap, linewidths=.5, cmap="YlGnBu")
plt.subplots(figsize=(11.7, 8.27))
sns.heatmap(heatmap, linewidths=.5, cmap="YlGnBu")
