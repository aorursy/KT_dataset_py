import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/phamdinhkhanh/AISchool/master/data_stocks.csv', 

                      header = 0, index_col = 0, sep = ',')

data.head()
pd.pivot_table(data, values = ['Open','High','Low','Close'], index = ['Symbols'], aggfunc = np.mean)
data['log'] = np.log(data['Close'])

data.head()
df = data.loc[:, ['Date','Symbols','log']]

df.head()
df1 = pd.pivot_table(df,values = ['log'], index = ['Date'], columns = ['Symbols'])

df1.diff()
df1.plot()
df1.describe