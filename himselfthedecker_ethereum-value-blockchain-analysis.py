import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn import preprocessing

pd.set_option('display.float_format', lambda x: '%.4f' % x)
dfBase = pd.read_csv('../input/EthereumBlockDifficultyGrowth.csv')
dfBase.rename(index=str, columns={'Value': 'Difficulty', 'Date(UTC)': 'Date_UTC'}, inplace=True)
dfBase = dfBase.assign(BlockRewards=pd.Series(pd.read_csv('../input/EthereumBlockRewardsChart.csv')['Value']).values)
dfBase = dfBase.assign(BlockSize=pd.Series(pd.read_csv('../input/EthereumBlockSizeHistory.csv')['Value']).values)
dfBase = dfBase.assign(HashRate=pd.Series(pd.read_csv('../input/EthereumNetworkHashRateGrowthRate.csv')['Value']).values)
dfBase = dfBase.assign(TotalTransactionFee=pd.Series(pd.read_csv('../input/EthereumTransactionFee.csv')['Value']).values)
dfBase = dfBase.assign(TransactionCount=pd.Series(pd.read_csv('../input/EthereumTransactionHistory.csv')['Value']).values)
dfBase = dfBase.assign(MarketCap=pd.Series(pd.read_csv('../input/EtherMarketCapChart.csv')['MarketCap']).values)
dfBase = dfBase.assign(PricePerUnit=pd.Series(pd.read_csv('../input/EtherMarketCapChart.csv')['Price']).values)
dfBase = dfBase.assign(SupplyOfCoins=pd.Series(pd.read_csv('../input/EtherMarketCapChart.csv')['Supply']).values)
dfBase = dfBase.assign(DailyGasUsage=pd.Series(pd.read_csv('../input/EthereumDailyGasUsedHistory.csv')['Value']).values)
dfBase = dfBase.assign(GasPrice=pd.Series(pd.read_csv('../input/EthereumGasPriceHistory.csv')['Value']).values)
dfBase.TotalTransactionFee = pd.to_numeric(dfBase.TotalTransactionFee.str.pad(dfBase.TotalTransactionFee.str.len().max(), 'left', '0').str.slice(0,18,1)) / (10**14)
dfBase.GasPrice = dfBase.GasPrice / (10**9)
dfBase.PricePerUnit = pd.to_numeric(dfBase.PricePerUnit.str.replace(',',''))
dfBase
df_Recent_Data = dfBase[(dfBase.UnixTimeStamp >= 1514764800)].copy()
df_Recent_Data.drop(['UnixTimeStamp'], axis=1, inplace=True)
sns.pairplot(df_Recent_Data)
df_Recent_Data.corr('pearson')
df_Recent_Normalized = df_Recent_Data.copy()
df_Recent_Normalized.set_index('Date_UTC', inplace=True)
min_max_scaler = preprocessing.MinMaxScaler()
df_Recent_Normalized[df_Recent_Normalized.columns] = min_max_scaler.fit_transform(df_Recent_Normalized[df_Recent_Normalized.columns])
df_Recent_Normalized
df_Recent_Normalized.corr('pearson')
traceDifficulty = go.Scatter(
                x = df_Recent_Normalized.index,
                y = df_Recent_Normalized.Difficulty,
                name = 'Difficulty (TH/solution)',
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=1.5)))

tracePricePerUnit = go.Scatter(
                x = df_Recent_Normalized.index,
                y = df_Recent_Normalized.PricePerUnit,
                name = 'Price Per Unit (USD)',
                mode = 'lines',
                marker = dict(color = 'rgba(82, 227, 68, 0.9)', line=dict(color='rgb(0,0,0)',width=0)))

layout = go.Layout(
    title='Difficulty (TH/solution) x Price Per ETH (USD)',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Normalized Value',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [traceDifficulty,tracePricePerUnit], layout=layout))
dfDiffPrice =  df_Recent_Data[['Date_UTC', 'Difficulty', 'PricePerUnit']]
dfDiffPrice.set_index('Date_UTC', inplace=True)
dfDiffPrice = dfDiffPrice.pct_change()

dfProfit = dfDiffPrice[(dfDiffPrice.PricePerUnit >= 0)]
dfLoss = dfDiffPrice[(dfDiffPrice.PricePerUnit <= 0)]

dataProfit = go.Histogram(
    x=dfProfit.Difficulty, 
    nbinsx = 100,
    opacity=0.75,
    name='Profits',
    marker=dict(
        color='rgba(82, 227, 68, 0.9)',
    )
)

dataLoss = go.Histogram(
    x=dfLoss.Difficulty, 
    nbinsx = 110,
    opacity=0.75,
    name='Losses',
    marker=dict(
        color='rgba(227, 68, 68)',
    )
)

layout = go.Layout(
    title='Histogram - Difficulty Variation from Previous Day in both Profits and Losses',
    xaxis=dict(
        title='Difficulty variation range'
    ),
    yaxis=dict(
        title='Occurences'
    ),
    bargap=0,
    bargroupgap=0,
    barmode='overlay'
)

fig = go.Figure(
    data=[dataProfit, dataLoss], 
    layout=layout
)
py.iplot(fig)
traceDifficulty = go.Scatter(
                x = df_Recent_Normalized.index,
                y = df_Recent_Normalized.Difficulty,
                name = 'Difficulty (TH/solution)',
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=1.5)))

traceHashRate = go.Scatter(
                x = df_Recent_Normalized.index,
                y = df_Recent_Normalized.HashRate,
                name = 'Hash Rate (TH/s)',
                mode = 'lines',
                marker = dict(color = 'rgba(168, 69, 227, 0.9)', line=dict(color='rgb(0,0,0)',width=0)))

layout = go.Layout(
    title='Difficulty (TH/solution) x Hash Rate (TH/s) x Price Per ETH (USD)',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Normalized Value',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [traceDifficulty,traceHashRate], layout=layout))
dfSupply =  df_Recent_Data[['Date_UTC', 'SupplyOfCoins']]
dfSupply.set_index('Date_UTC', inplace=True)
dfSupply.pct_change()
traceTransactionCount = go.Scatter(
                x = df_Recent_Normalized.index,
                y = df_Recent_Normalized.TransactionCount,
                name = 'Transaction Volume',
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=1.5)))

tracePricePerUnit = go.Scatter(
                x = df_Recent_Normalized.index,
                y = df_Recent_Normalized.PricePerUnit,
                name = 'Price Per Unit (USD)',
                mode = 'lines',
                marker = dict(color = 'rgba(82, 227, 68, 0.9)', line=dict(color='rgb(0,0,0)',width=0)))

layout = go.Layout(
    title='Transaction Volume (Trans/day) x Price Per ETH (USD)',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Normalized Value',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [traceTransactionCount, tracePricePerUnit], layout=layout))