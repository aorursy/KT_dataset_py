import pandas as pd



df = pd.read_csv("../input/stock-index/stock_exchange_index.csv")

df.head()
df.apply(lambda x: x.isna().sum())