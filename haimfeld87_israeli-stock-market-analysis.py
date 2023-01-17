import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from datetime import datetime
df=pd.read_csv("../input/Israel_Stocks.csv")
df.columns
df.head()
stocks_names=list(df["Symbol"].str.split(' ', expand=True).stack().unique())
print("There are " + str(len(stocks_names)) + " stocks in the dataset.")
time_format = '%d/%m/%y'
for z in stocks_names:
    i=df[df["Symbol"]==z]
    time = [datetime.strptime(x, time_format) for x in i['Date']]
    plt.plot(time,i['Closing Price (0.01 NIS)'])
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(z)
    plt.show()