import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
banknifty = pd.read_csv("../input/banknifty.csv",parse_dates=['date','time'])

nifty50 = pd.read_csv("../input/nifty50.csv",parse_dates=['date','time'])

nifty50.head()
banknifty.head()
banknifty = banknifty.drop(['index'],axis=1)

nifty50 = nifty50.drop(['index'],axis=1)
nifty50.describe()
nifty50.loc[nifty50['high']==9119.2]
nifty50.loc[nifty50['high']==5127.25]
nifty50_mean = nifty50.groupby('date').mean()

nifty50_mean.head()
nifty50_mean.plot(figsize=(12,8))
banknifty.describe()
banknifty.loc[banknifty['high']==20907.550]
banknifty.loc[banknifty['high']==1407.050]
banknifty_mean = banknifty.groupby('date').mean()

banknifty_mean.head()
banknifty_mean.plot(figsize=(12,8))
maximum_drop_nifty50 =  max(nifty50['open'] - nifty50['low'])

maximum_drop_nifty50
nifty50.loc[(nifty50['open'] - nifty50['low']) == maximum_drop_nifty50]
maximum_up_nifty50 =  max(nifty50['high'] - nifty50['open'])

nifty50.loc[(nifty50['high'] - nifty50['open']) == maximum_up_nifty50]
maximum_drop_banknifty =  max(banknifty['open'] - banknifty['low'])

maximum_drop_banknifty
banknifty.loc[(banknifty['open'] - banknifty['low']) == maximum_drop_banknifty]
maximum_up_banknifty =  max(banknifty['high'] - banknifty['open'])

banknifty.loc[(banknifty['high'] - banknifty['open']) == maximum_up_banknifty]
sns.heatmap(banknifty.corr(),annot=True)
nifty50['high'].plot('kde')
banknifty['high'].plot('kde')