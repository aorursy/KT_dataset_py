# import library

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# import data

df = pd.read_csv("/kaggle/input/stocknews/upload_DJIA_table.csv")
# looking at the top five rows of the data

df.head()
# looking at the bottom five rows of the data

df.tail()
df.info()
df.shape
df.size
# dataframe

df.ndim
# for series

df['Date'].ndim
df.index
df.columns
df.count()
df.sum()
df.cumsum().head()
df.min()
df.max()
df['Open'].idxmin()
df['Open'].idxmax()
df.describe()
df.mean()
df.median()
df.quantile([0.25,0.75])
df.var()
df.std()
df.cummax().head()
df.cummin().head()
df['Open'].cumprod().head()
len(df)
df.isnull().head()
df.corr()