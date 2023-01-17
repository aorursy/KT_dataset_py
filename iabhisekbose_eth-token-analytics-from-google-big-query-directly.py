import pandas as pd

import plotly as py

import plotly.graph_objs as go

from plotly.tools import FigureFactory as FF

from pandas.io import gbq

    

    # Perform a query.

#QUERY = ('SELECT * FROM `ethchain-216308.EthData.ERC721Data` LIMIT 1000')



QUERY = (

    'SELECT contracts.address,COUNT(1) AS tx_count FROM `bigquery-public-data.ethereum_blockchain.contracts` AS contracts JOIN `bigquery-public-data.ethereum_blockchain.transactions` AS transactions ON (transactions.to_address = contracts.address)'

    'GROUP BY contracts.address ORDER BY tx_count DESC LIMIT 100'

)



#Visualizing Etherwise date & volume

QUERYEthTransVol = (

'SELECT date(block_timestamp) as date,sum(value)/power(10,18) as volume FROM `bigquery-public-data.ethereum_blockchain.transactions` group by date order by date'

)



#Visualizing average Ether costs over time

queryCost = """

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

HAVING tx_date >= '2019-01-01' AND tx_date <= '2019-02-25'

ORDER BY tx_date

"""



top10_story_df = gbq.read_gbq(QUERY, project_id='ethchain-216308',dialect='standard')



tran_df = gbq.read_gbq(QUERYEthTransVol, project_id='ethchain-216308',dialect='standard')



tran_dfForCost = gbq.read_gbq(queryCost, project_id='ethchain-216308',dialect='standard')



# Create a table figure from the DataFrame

#top10_story_figure = plotly.figure_factory.create_table(top10_story_df)



py.offline.init_notebook_mode(connected=True)



py.offline.iplot({

    "data": [go.Bar(x=top10_story_df['address'], y=top10_story_df['tx_count'])],

    "layout": go.Layout(title="Addresswise ETH Token Count - Vertical Bars",hovermode= "closest",xaxis={"type": "category","title":"Eth Address"},yaxis={"title":"tx_count"})

}, filename='Addresswise-ETH-Token-Count-verticals.html')



#top10_story_df



trace = go.Scatter(

    x=top10_story_df['address'],

    y=top10_story_df['tx_count'],

    mode='lines',

    text=top10_story_df['address'],

    

)



layout = go.Layout(

    title='Addresswise ETH Token Count - Scatter',

    xaxis=dict(

        title="Eth Address"

    ),

    yaxis=dict(

        title="tx_Count"

    )

)



data = [trace]



fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig, filename='Addresswise-ETH-Token-Count-scatter.html')



#tran_df



py.offline.iplot({

    "data": [go.Bar(x=tran_df['date'], y=tran_df['volume'])],

    "layout": go.Layout(title="Datewise Ethereum transaction volume - Vertical Bars",hovermode= "closest",xaxis={"title":"Date"},yaxis={"title":"Volume"})

},filename='Datewise-ETH-Volume.html')



#tran_dfForCost

trace1 = go.Scatter(

    x=tran_dfForCost['tx_date'],

    y=tran_dfForCost['sum_tx_ether'],

    mode='lines+markers'

    

)

trace2 = go.Scatter(

    x=tran_dfForCost['tx_date'],

    y=tran_dfForCost['avg_tx_gas_cost'],

    mode='lines+markers'

    

)



layout1 = go.Layout(

    title='Visualizing Sum Ether costs over time-2019-Scatter',

    xaxis=dict(

        title="Date"

    ),

    yaxis=dict(

        title="Sum Ether Cost"

    )

)



data1 = [trace1,trace2]



fig1 = go.Figure(data=data1, layout=layout1)

py.offline.iplot(fig1,filename='Datewise-ETH-GasCost.html')







py.offline.iplot({

    "data": [go.Bar(x=tran_dfForCost['tx_date'], y=tran_dfForCost['avg_tx_gas_cost'])],

    "layout": go.Layout(title="Visualizing average Ether costs over time-2019-Bar",xaxis={"title":"Date"},yaxis={"title":"Avg Gas Cost"})

},filename='Datewise-ETH-GasCost.html')
