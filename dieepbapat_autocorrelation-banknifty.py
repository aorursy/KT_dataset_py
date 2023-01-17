import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



plt.rcParams['figure.figsize']=(20,8)

data = pd.read_csv("../input/stock-market-india/FullDataCsv/NIFTY_BANK__EQ__INDICES__NSE__MINUTE.csv")

data.shape
data.head()
data['ret']=data.close.pct_change()
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(data.ret.dropna())

plt.ylim((-0.1,0.1))

plt.show()


from statsmodels.graphics.tsaplots import plot_pacf



plot_pacf(data.ret.dropna(), lags=150)

plt.ylim((-0.1,.1))

plt.show()

data.tail()