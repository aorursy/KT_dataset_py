import pandas as pd

import seaborn as sns

from fbprophet import Prophet

df = pd.read_csv('../input/brent-oil-prices/BrentOilPrices.csv')

df.info()

df.head()
df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y") #format date data to appropriate format

df.head()
sns.lineplot(x='Date', y='Price', data=df)
#Standard procedure is to rename date column to DS:

df.rename(columns={"Date": "ds", "Price": "y"}, inplace=True)
df = df[df['ds'] > '2004-01-01']
m = Prophet()

m.fit(df)

future = m.make_future_dataframe(periods=365) #forecasting 365 days in future

forecast = Prophet(interval_width=0.95).fit(df).predict(future)

m.plot(forecast)
dfn = forecast.set_index('ds')[['yhat']].join(df.set_index('ds'))

dfn = dfn[dfn.index > '2016-01-01']
ax = sns.lineplot(data = dfn)

sns.set_style('darkgrid')

sns.set_palette('rainbow')
dftd = pd.read_csv('../input/brent-oil-td/RBRTEd.csv')

dftd['Date'] = pd.to_datetime(dftd['Date'], format="%b %d, %Y") #format date data to appropriate format

dftd.rename(columns={"Date": "ds", "Price": "Price TD"}, inplace=True)

dfx = dfn.join(dftd.set_index('ds'))
df['ds'].max() #last record of original df to segregate predicted from actual
from matplotlib import pyplot as plt

a = dfx['y']

b = dfx['yhat']

c = dfx['Price TD']

c = dfx[dfx.index > '2019-09-30']



plt.plot(a)

plt.plot(b)

plt.plot(c)

plt.show()