from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

import pandas as pd



read_data = pd.read_csv("../input/bitcoin_price_Training - Training.csv")

clf = RandomForestRegressor()

clf.fit(read_data[['Open', 'High', 'Low']], read_data["Close"][::-1])

test_data = pd.read_csv("../input/bitcoin_price_1week_Test - Test.csv")

plt.plot(clf.predict(test_data[['Open', 'High', 'Low']]))