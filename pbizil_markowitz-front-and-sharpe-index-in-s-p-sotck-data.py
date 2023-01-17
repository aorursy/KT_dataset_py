import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pandas_datareader.data as web

from datetime import datetime

import seaborn as sns



data = pd.read_csv('../input/sandp500/all_stocks_5yr.csv', index_col="date")

data.describe()
def orgDataTop5assets(dataframe):

    date_start = min(dataframe.index)

    date_end = max(dataframe.index)

    assets = dataframe["Name"].unique()

    assets_top5 = pd.DataFrame(index=assets, columns=["Open_Start", "Open_End", "Spread"])

    assets_top5 = assets_top5.drop(['PCLN', 'GOOG'])



    for a in assets:

        df = dataframe[dataframe["Name"]==a]

        dates = df.index

        assets_top5["Open_Start"].loc[a] = df[df["Name"]==a].loc[min(df.index)]["open"]

        assets_top5["Open_End"].loc[a] = df[df["Name"]==a].loc[max(df.index)]["open"]

    

    assets_top5['Spread']=assets_top5["Open_End"]-assets_top5["Open_Start"]

    assets_top5 = assets_top5.sort_values(["Spread"], ascending=False).head(5)

    return assets_top5



assets = orgDataTop5assets(data).reset_index()["index"].unique()
def portfolio(dataframe):

    ptf = pd.DataFrame(index=dataframe.index.unique(),columns=assets)

    dates = dataframe.index.unique()



    for a in assets:

        df = dataframe[dataframe["Name"]==a]

        for d in dates:

            ptf[a].loc[d] = df["open"].loc[d]

        

    ptf=ptf.astype(float)

    return ptf 

    
ptf = portfolio(data)

((ptf / ptf.iloc[0])*100).plot(figsize=(15,6))

plt.show()
corr = ptf.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
return_daily = ptf.pct_change()

return_yearly = return_daily.mean() * 250

cov_daily = return_daily.cov()

cov_yearly = cov_daily * 250
return_wallet = []

weights_assets = []

volatility_wallet = []

sharpe_ratio = []

num_assets = len(assets)

num_wallets = 100000

np.random.seed(101)



for wallet in range(num_wallets):

    #setting random weights

    weight = np.random.random(num_assets)

    weight /= np.sum(weight)

    #calculation of portfolio return

    returns = np.dot(weight, return_yearly)

    #portfolio volatility calculation

    volatility = np.sqrt(np.dot(weight.T, np.dot(cov_yearly, weight)))

    #sharpe index

    sharpe = returns / volatility

    

  

    sharpe_ratio.append(sharpe)

    return_wallet.append(returns)

    volatility_wallet.append(volatility)

    weights_assets.append(weight)



wallet = {'Return': return_wallet,

            'Volatility': volatility_wallet,

             'Sharpe Ratio': sharpe_ratio}



for count,asset in enumerate(assets):

    wallet[asset+' Weight'] = [Weight[count] for Weight in weights_assets]



df = pd.DataFrame(wallet)

columns = ['Return', 'Volatility', 'Sharpe Ratio'] + [asset+' Weight' for asset in assets]

df = df[columns]

lower_volatility = df['Volatility'].min()

best_sharpe = df['Sharpe Ratio'].max()

wallet_sharpe = df.loc[df['Sharpe Ratio'] == best_sharpe]

wallet_min_variance = df.loc[df['Volatility'] == lower_volatility]
plt.style.use('seaborn-dark')

df.plot.scatter(x='Volatility', y='Return', c='Sharpe Ratio',

                cmap='RdYlGn', edgecolors='black', figsize=(25, 7), grid=True)

plt.scatter(x=wallet_sharpe['Volatility'], y=wallet_sharpe['Return'], c='red', marker='o', s=200)

plt.scatter(x=wallet_min_variance['Volatility'], y=wallet_min_variance['Return'], c='blue', marker='o', s=200 )

plt.xlabel('Volatility')

plt.ylabel('Expected Return')

plt.title('S&P 500 - Markowitz Frontier')

plt.show()
print("This is a Minimum Variance Wallet:", '\n', wallet_min_variance.T)

print("This is a Minimum with the best Sharpe Ratio:", '\n', wallet_sharpe.T)