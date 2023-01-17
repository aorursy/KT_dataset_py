# import library

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# import data

df = pd.read_csv('/kaggle/input/stocknews/upload_DJIA_table.csv')
df.head()
df.loc[:, 'Open']
df['Open']
df.Open
df.loc[:, ['Open', 'Close']]
df[['Open', 'Close']]
df.loc[0]
df.loc[[0,1,10]]
df.loc[0, 'Open']
df.loc[1, ['Open', 'Close']]
df.loc[[0, 1], ['Open']]
df.loc[[0, 1], ['Open', 'Close']]
df.iloc[0]
df.iloc[:, 5]
df.iloc[0, 0]
df.iloc[[0, 1, 3], [0, 1]]
df.nlargest(3,'Open') 
df.nsmallest(3,'Open') 
df.sample(3)
df.sample(frac = 0.3)
df[df.Open >= 18281.949219]
df.loc[1:5, :]
df.loc[:, 'Open' : 'Close']
df.loc[1:3, 'Open' : 'Close']
df.iloc[0:2, :]
df.iloc[:, 1:4]
df.iloc[0:2, 0:2]
df.iloc[:2, :2]
df.Open == 17355.210938
df[df.Open == 17355.210938]
df.loc[df.Open == 17355.210938]
df[df.Open.isin([17355.210938])]
df[(df.Open == 17355.210938) | (df.Close == 17949.369141)]
# row 0 and 2 contains these values so it is not in output

df[~df.Open.isin([17924.240234, 17456.019531])]
df.loc[:, df.isin([17456.019531]).any()]
df.filter(items=['Open', 'Close'])
df.filter(like="2", axis=0)
df.filter(regex="[^OpenCloseHigh]")
df[(df['Open'] > 18281.949219) & (df['Date'] > '2015-05-20')]