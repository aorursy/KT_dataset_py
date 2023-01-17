import pandas as pd

import numpy as np

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
df = pd.read_csv('../input/GFDDData.csv',usecols=['Country Name','Indicator Name','2017'])
df.head(5)
df.shape
df.columns
df.dtypes
pd.unique(df['Indicator Name'])
len(pd.unique(df['Indicator Name']))
piv = pd.pivot_table(data=df, index='Country Name',columns='Indicator Name',values='2017')

piv.head()
fillter = piv['Banking crisis dummy (1=banking crisis, 0=none)'] == 1

fillter.head(3)
piv['Banking crisis dummy (1=banking crisis, 0=none)']
b = piv['Banking crisis dummy (1=banking crisis, 0=none)'].where(fillter).dropna()

b
piv.isna().sum().sort_values(ascending=True).head(30)
piv['Bank concentration (%)']
piv['Bank concentration (%)'].hist(bins=20,figsize=(16,6))
plt.hist(piv['Bank concentration (%)'],bins=20)
most_values_countrys = df.dropna().groupby(by='Country Name').count().sort_values(by='2017',ascending=False).head(5)

most_values_countrys

list(most_values_countrys.index)
ndf = pd.DataFrame(piv.loc[list(most_values_countrys.index)])

ndf
#A DataFrame object has two axes: “axis 0” and “axis 1”. “axis 0” represents rows and “axis 1” represents columns. 

ndf.dropna(axis=1)
piv.fillna(piv.mean())
extract = piv.fillna(piv.mean())

extract
extract.to_csv("file.csv")
new_ext = pd.melt(extract)

new_ext
new_ext.to_csv(encoding="utf-8",decimal=';end;') 