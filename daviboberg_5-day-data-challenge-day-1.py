import pandas as pd
data = pd.read_csv("../input/CryptocoinsHistoricalPrices.csv", index_col=0)

data.head()
data.describe()
data["coin"].value_counts()
btc = data[data["coin"]=="BTC"]

btc.describe()
data[data["coin"]=="ETH"].describe()