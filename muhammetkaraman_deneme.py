# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Country.csv")

data.head()
data.info()
data.corr()
numeric_data = data.copy()


for each in numeric_data.columns:
    if numeric_data[each].dtype != "float64":
        del numeric_data[each]

numeric_data.head()
numeric_data = pd.concat([numeric_data,data[["ShortName"]]],axis=1)
numeric_data.head()
f,ax = plt.subplots(1,3,figsize=(30,10))
numeric_data.LatestIndustrialData.plot(kind="hist",ax=ax[0],color="red",grid=True,alpha=0.5)
ax[0].set_xlabel("LatestIndustrialData",fontsize=30)
ax[0].legend(loc="upper left")
numeric_data.LatestTradeData.plot(kind="hist",ax=ax[1],color="green",grid=True,alpha=0.5)
ax[1].set_xlabel("LatestTradeData",fontsize=30)
ax[1].legend(loc="upper left")
numeric_data.LatestWaterWithdrawalData.plot(kind="hist",ax=ax[2],grid=True,alpha=0.5)
ax[2].set_xlabel("LatestWaterWithdrawalData",fontsize=30)
ax[2].legend(loc="upper left")


plt.show()
f,ax = plt.subplots(3,1,figsize=(10,30))
numeric_data.plot(kind="scatter",x="LatestIndustrialData",y="LatestTradeData",ax=ax[0],color="red",grid=True)

numeric_data.plot(kind="scatter",x="LatestIndustrialData",y="LatestWaterWithdrawalData",ax=ax[1],color="green",grid=True)

numeric_data.plot(kind="scatter",x="LatestTradeData",y="LatestWaterWithdrawalData",ax=ax[2],grid=True)
numeric_data.plot(kind="line",figsize=(15,15),grid=True)