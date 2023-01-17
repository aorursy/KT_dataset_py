import pandas as pd

df = pd.read_csv("../input/chennai_reservoir_rainfall.csv",parse_dates=["Date"])

df.head()
df.info()
expected_days = df.Date.max()-df.Date.min()

print(expected_days.days)

actual_days = df.shape[0]

print(actual_days)
import matplotlib.pyplot as plt

fig,ax =plt.subplots(figsize=(18,6))

df.plot(kind ="line",x="Date",y="POONDI",ax=ax)
zoom_range = df[(df.Date >= '2010-01-01') & (df.Date < '2011-01-01')].index

fig,ax = plt.subplots(figsize=(18,6))

df.loc[zoom_range].plot(kind="line",x="Date",y="POONDI",ax=ax)
from statsmodels.tsa.seasonal import seasonal_decompose

defreq = 12

model = "additive"

decomposition = seasonal_decompose(

    df.set_index("Date").POONDI.interpolate("linear"),

    freq = defreq,

    model=model)
trend = decomposition.trend

sesonal = decomposition.seasonal

resid = decomposition.resid
fig,ax =plt.subplots(figsize=(18,6))

df.plot(kind ="line",x="Date",y="POONDI",ax=ax)

trend.plot(ax=ax,label="trend")
mon = df.Date.dt.to_period("M")

mon_group = df.groupby(mon)

df_monthly = mon_group.sum()

df_monthly.head()
import matplotlib.pyplot as plt

fig,ax =plt.subplots(figsize=(18,6))

df_monthly.plot(kind ="line",y="POONDI",ax=ax)
from statsmodels.tsa.seasonal import seasonal_decompose

defreq = 12

model = "additive"

decomposition = seasonal_decompose(

    df_monthly.POONDI.interpolate("linear"),

    freq = defreq,

    model=model)
trend = decomposition.trend

sesonal = decomposition.seasonal

resid = decomposition.resid
fig,ax =plt.subplots(figsize=(18,6))

df_monthly.plot(kind ="line",y="POONDI",ax=ax,label="Poondi")

trend.plot(ax=ax,label="Trend")

plt.legend(loc='upper left')

fig,ax =plt.subplots(figsize=(18,6))

df_monthly.plot(kind ="line",y="POONDI",ax=ax,label="Poondi")

sesonal.plot(ax=ax,label="Seasonality")

plt.legend(loc='upper left')
fig,ax =plt.subplots(figsize=(18,6))

df_monthly.plot(kind ="line",y="POONDI",ax=ax,label="Poondi")

resid.plot(ax=ax,label="Residual")

plt.legend(loc='upper left')
year = df.Date.dt.to_period("Y")

year_group = df.groupby(year)

df_yearly = year_group.sum()

df_yearly.head()
import matplotlib.pyplot as plt

fig,ax =plt.subplots(figsize=(18,6))

df_yearly.plot(kind ="line",y="POONDI",ax=ax)
from statsmodels.tsa.seasonal import seasonal_decompose

defreq = 12

model = "additive"

decomposition = seasonal_decompose(

    df_yearly.POONDI.interpolate("linear"),

    freq = defreq,

    model=model)
trend = decomposition.trend

sesonal = decomposition.seasonal

resid = decomposition.resid
fig,ax =plt.subplots(figsize=(18,6))

df_yearly.plot(kind ="line",y="POONDI",ax=ax,label="Poondi")

trend.plot(ax=ax,label="Trend")

plt.legend(loc='upper left')

fig,ax =plt.subplots(figsize=(18,6))

df_yearly.plot(kind ="line",y="POONDI",ax=ax,label="Poondi")

sesonal.plot(ax=ax,label="Seasonality")

plt.legend(loc='upper left')
df_monthly.index.quarter
df_monthly.head()
df_monthly.info()
df_monthly.index
df_monthly[df_monthly.index > '2018-01']
train = df_monthly[df_monthly.index < '2018-01']

test = df_monthly[df_monthly.index >= '2018-01']
fig, ax = plt.subplots(figsize=(18,6))

train.plot(y="POONDI", ax=ax, label="train")

test.plot(y="POONDI", ax=ax, label="test")

plt.legend(loc='upper left')
#df1 = df.set_index('Date')



df_quaterly = df.set_index('Date').resample('QS').sum()
df_quaterly
import matplotlib.pyplot as plt

fig,ax =plt.subplots(figsize=(18,6))

df_quaterly.plot(kind ="line",y="POONDI",ax=ax)
from statsmodels.tsa.seasonal import seasonal_decompose

defreq = 12

model = "additive"

decomposition = seasonal_decompose(

    df_quaterly.POONDI.interpolate("linear"),

    freq = defreq,

    model=model)
trend = decomposition.trend

sesonal = decomposition.seasonal

resid = decomposition.resid
fig,ax =plt.subplots(figsize=(18,6))

df_quaterly.plot(kind ="line",y="POONDI",ax=ax,label="Poondi")

trend.plot(ax=ax,label="Trend")

plt.legend(loc='upper left')
fig,ax =plt.subplots(figsize=(18,6))

df_quaterly.plot(kind ="line",y="POONDI",ax=ax,label="Poondi")

sesonal.plot(ax=ax,label="Seasonality")

plt.legend(loc='upper left')