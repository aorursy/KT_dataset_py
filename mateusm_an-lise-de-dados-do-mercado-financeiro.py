import pandas_datareader.data as data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline

from datetime import datetime
start = datetime(2006, 1, 1)

end = datetime(2016, 1, 1)



BAC = data.DataReader("BAC", "yahoo", start, end) # Bank of America

C = data.DataReader("C", "yahoo", start, end) # CitiGroup

GS = data.DataReader("GS", "yahoo", start, end) # Goldman Sachs

JPM =data.DataReader("JPM", "yahoo", start, end) # JPMorgan Chase

MS = data.DataReader("MS", "yahoo", start, end) # Morgan Stanley

WFC = data.DataReader("WFC", "yahoo", start, end) # Wells Fargo

BAC.head()
tickers = ['JPM', 'BAC', 'WFC','GS', 'MS', 'C']

tickers = sorted(tickers)
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1,keys=tickers)

bank_stocks.head()
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
bank_stocks.head()
bank_stocks.xs('Close', level='Stock Info', axis=1).max()
returns = pd.DataFrame()
for tick in tickers:

    returns[tick + 'Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()
returns['CReturn'].plot()

plt.title(' Retorno das ações do Citigroup')
sns.pairplot(returns[1:])
returns.idxmax() # pega pelo id
returns.idxmin()
returns.std()
by_date = returns.copy()

by_date.reset_index(inplace = True)

by_date['Date'] = pd.to_datetime(by_date['Date'])

by_date['Year'] = by_date['Date'].apply(lambda time: time.year) # Criando coluna com o ano de cada ação

by_date[by_date['Year'] == 2014].std() #Pegando desvio padrão
# outra alternativa:

'''

returns[(returns.index.date >= datetime.date(2015,1,1)) & (returns.index.date < datetime.date(2016,1,1))].std()

'''
sns.distplot(by_date[by_date['Year'] == 2014]['MSReturn'])
sns.distplot(by_date[by_date['Year'] == 2010]['CReturn'])
for tick in tickers:

    bank_stocks[tick]['Close'].plot(figsize=(12,4), label=tick)

    # ou bank_stocks.xs(key='Close', level='Stock Info', axis=1).plot()

plt.legend()
plt.figure(figsize=(12,6))

C2008 = C['Close'][C.index.year == 2008]

C2008_mm = C2008.rolling(30).mean().plot(label = 'Média móvel de 30 dias')

C2008.plot(label = 'Fechamento')

plt.legend()

plt.title("Média movel e fechamento das ações do CitiGroup no ano de 2008")
corr = bank_stocks.xs(key='Close', level='Stock Info', axis=1).corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr, cmap='coolwarm', annot=True)
sns.clustermap(corr, cmap='coolwarm', annot=True)