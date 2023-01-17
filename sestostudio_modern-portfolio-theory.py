# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats.mstats import gmean

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
price_df = pd.read_csv('../input/nyse/prices.csv')

sec_df = pd.read_csv('../input/nyse/securities.csv')

fund_df = pd.read_csv('../input/nyse/fundamentals.csv')
plt.figure(figsize=(15, 6))

ax = sns.countplot(y='GICS Sector', data=sec_df)

plt.xticks(rotation=45)
price_df.head()
price_df.isna().sum()
sec_df = sec_df.rename(columns = {'Ticker symbol' : 'symbol','GICS Sector' : 'sector'})

sec_df.head()
price_df  = price_df.merge(sec_df[['symbol','sector']], on = 'symbol')

price_df['date'] = pd.to_datetime(price_df['date'])

price_df.head()
price_df = price_df[price_df['date'] >= '2016-01-01']
sector_pivot = pd.pivot_table(price_df, values = 'close', index = ['date'],columns = ['sector']).reset_index()

sector_pivot
plt.figure(figsize = (10,10))

sns.heatmap(sector_pivot.corr(),annot=True, cmap="coolwarm")
price_df['return'] = np.log(price_df.close / price_df.close.shift(1)) + 1

price_df['good'] = price_df['symbol'] == price_df['symbol'].shift(1)

price_df = price_df.drop(price_df[price_df['good'] == False].index)

price_df.dropna(inplace = True)
risk_free = 0.032

sector_df = pd.DataFrame({'return' : (price_df.groupby('sector')['return'].mean() - 1) * 252, 'stdev' : price_df.groupby('sector')['return'].std()})

sector_df['sharpe'] = (sector_df['return'] - risk_free) / sector_df['stdev']

plt.figure(figsize = (12,8))

ax = sns.barplot(x= sector_df['sharpe'], y = sector_df.index)
port_list = sector_df[sector_df['sharpe'] >= 1].index

port_list
price_df.head()
port_stock = []

return_stock = []

def get_stock(sector):

    list_stocks = price_df[price_df['sector'] == sector]['symbol'].unique()

    performance = price_df.groupby('symbol')['return'].apply(lambda x : (gmean(x) - 1) * 252).sort_values(ascending = False)

    

    for i in range(len(performance)):

        if performance.index[i] in list_stocks:

            port_stock.append(performance.index[i])

            return_stock.append(performance[i])

            break

    

for sector in port_list:

    get_stock(sector)



return_stock
port_df = price_df[price_df['symbol'].isin(port_stock)].pivot('date','symbol','return')

return_pred = []

weight_pred = []

std_pred = []

for i in range(1000):

    random_matrix = np.array(np.random.dirichlet(np.ones(len(port_stock)),size=1)[0])

    port_std = np.sqrt(np.dot(random_matrix.T, np.dot(port_df.cov(),random_matrix))) * np.sqrt(252)

    port_return = np.dot(return_stock, random_matrix)

    return_pred.append(port_return)

    std_pred.append(port_std)

    weight_pred.append(random_matrix)
pred_output = pd.DataFrame({'weight' : weight_pred , 'return' : return_pred, 'stdev' :std_pred })

pred_output['sharpe'] = (pred_output['return'] - risk_free) / pred_output['stdev']

pred_output.head()
max_pos = pred_output.iloc[pred_output.sharpe.idxmax(),:]

safe_pos = pred_output.iloc[pred_output.stdev.idxmin(),:]
plt.subplots(figsize=(15,10))

#ax = sns.scatterplot(x="Stdev", y="Return", data=pred_output, hue = 'Sharpe', size = 'Sharpe', sizes=(20, 200))



plt.scatter(pred_output.stdev,pred_output['return'],c=pred_output.sharpe,cmap='OrRd')

plt.colorbar()

plt.xlabel('Volatility')

plt.ylabel('Return')



plt.scatter(max_pos.stdev,max_pos['return'],marker='^',color='r',s=500)

plt.scatter(safe_pos.stdev,safe_pos['return'],marker='<',color='g',s=500)

#ax.plot()
print("The highest sharpe porfolio is {} sharpe, at {} volitality".format(max_pos.sharpe.round(3),max_pos.stdev.round(3)))



for i in range(len(port_stock)):

    print("{} : {}%".format(port_stock[i],(max_pos.weight[i] * 100).round(3)))
print("The safest porfolio is {} risk, {} sharpe".format(safe_pos.stdev.round(3), safe_pos.sharpe.round(3)))

for i in range(len(port_stock)):

    print("{} : {}%".format(port_stock[i],(safe_pos.weight[i] * 100).round(3)))