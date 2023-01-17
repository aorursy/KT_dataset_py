import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
pd.read_csv("../input/apple-stock-19802014/appl_1980_2014.csv")
apple = pd.read_csv("../input/apple-stock-19802014/appl_1980_2014.csv")
apple.dtypes
apple.Date = pd.to_datetime(apple.Date)
apple.set_index("Date")
x = apple[apple.duplicated("Date")]

if len(x) != 0:

    print("Yes there are duplicates in date column")

else:

    print("No there are no duplicates in date column")
apple = apple.sort_values(by="Date",ascending=True).reset_index(drop=True)

apple
apple["month"] = pd.DatetimeIndex(apple.Date).month

apple["Date_wo"] = pd.DatetimeIndex(apple.Date).day

apple.groupby(by="month")[["Date_wo"]].max().reset_index()
differnce = apple.Date.max() - apple.Date.min() 

str(differnce)
months_data = apple["month"].count()

months_data
plt.figure(figsize=(13.5, 9))

plt.hist(apple["Adj Close"])

plt.show()