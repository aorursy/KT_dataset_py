

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.xlsx',na_values=["ND"])

# df = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv') # has extra col

print(df.shape[0])

df.head(12)
df.dropna(thresh=3,inplace=True)

print(df.shape[0])

df.head(12)
df.columns = df.columns.str.strip()

df.columns = df.columns.str.replace(r"/US\$","_USD",regex=True)

df.rename(columns={'Time Serie':"date"},inplace=True)

df.head()
df.to_csv('daily_USD_Foreign_Exchange_Rates.csv.gz',index=False,compression="gzip")