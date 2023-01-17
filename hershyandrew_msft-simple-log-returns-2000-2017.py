import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_datareader import data as wb

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt
MSFT = pd.read_csv("../input/MSFT_2000.csv", index_col = 'Date')

MSFT.head()
MSFT["simple_return"] = (MSFT["MSFT"] / MSFT["MSFT"].shift(1)) - 1

print (MSFT["simple_return"])
MSFT['simple_return'].plot(figsize=(15, 10))

plt.show()
daily_return = MSFT['simple_return'].mean()

print ("Daily Return Average is: "+str(round(daily_return,4)*100)+ "%")



yearly_return = MSFT['simple_return'].mean()*250

print ("Yearly Return Average is: "+str(round(yearly_return,4)*100)+ "%")
MSFT["log_return"] = np.log(MSFT["MSFT"]/MSFT["MSFT"].shift(1))

print (MSFT["log_return"])
daily_return_log = MSFT['log_return'].mean()

print ("Daily Log Return Average is: "+str(round(daily_return_log,4)*100)+ "%")



yearly_return_log = MSFT['log_return'].mean()*250

print ("Yearly Log Return Average is: "+str(round(yearly_return_log,4)*100)+ "%")