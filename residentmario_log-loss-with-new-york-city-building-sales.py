from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
import pandas as pd

import numpy as np

sales = pd.read_csv("../input/nyc-rolling-sales.csv", index_col=0)

sales.head(3)
df = sales[['SALE PRICE', 'TOTAL UNITS']].dropna()

df['SALE PRICE'] = df['SALE PRICE'].str.strip().replace("-", np.nan)

df = df.dropna()



X = df.loc[:, 'TOTAL UNITS'].values[:, np.newaxis].astype(float)

y = df.loc[:, 'SALE PRICE'].astype(int) > 1000000



from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X[:1000], y[:1000])

y_hat = clf.predict(X)
from sklearn.metrics import log_loss

log_loss(y, y_hat)