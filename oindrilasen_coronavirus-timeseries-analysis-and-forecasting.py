# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

# Plots

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns



#

import warnings

warnings.filterwarnings("ignore")

import itertools

import statsmodels.api as sm

#

from statsmodels.tsa.arima_model import ARIMA
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.options.display.float_format = '{:,.0f}'.format
df1 = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200201.csv")

df2 = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200205.csv")

df3 = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv")

df4 = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200130.csv")

df5 = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200127.csv")

df6 = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv")

df7 = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200128.csv")

df_summary = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126 - SUMMARY.csv")

df_clean = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")
print(df1.shape)

print(df2.shape)

print(df3.shape)

print(df4.shape)

print(df5.shape)

print(df6.shape)

print(df7.shape)

print(df_summary.shape)

print(df_clean.shape)
df_covid = pd.concat([df1,df2,df3,df4,df5,df6,df7], axis=0)

df_covid = df_covid.sort_values("Last Update")

df_covid.shape
df_covid.info()
df_covid.head()
df_covid.rename(columns = {'Province/State':'State'}, inplace = True) 

df_covid.rename(columns = {'Country/Region':'Country'}, inplace = True) 

df_covid.rename(columns = {'Last Update':'last_update'}, inplace = True) 
df_covid.last_update = pd.to_datetime(df_covid.last_update).dt.date.astype(str)
df_covid.isnull().sum()
df_covid["Confirmed"] = df_covid.Confirmed.fillna(0)

df_covid["Suspected"] = df_covid.Suspected.fillna(0)

df_covid["Recovered"] = df_covid.Recovered.fillna(0)

df_covid["Death"] = df_covid.Death.fillna(0)
df_covid.head(2)
df_dates = df_covid.groupby("last_update")[["last_update","Confirmed","Suspected","Recovered","Death"]].sum().reset_index()
df_dates['last_update'].min(), df_dates['last_update'].max()
plt.figure(figsize=(15,6))

sns.set_color_codes("pastel")

pl1 = sns.barplot(x="last_update", y=df_dates.sort_index().Confirmed.cumsum(), data=df_dates,

            label="Total", color="b").set_title("Cumulative Confirmed cases of Covid-19")

plt.xticks(rotation=90,horizontalalignment='right',

    fontweight='light',

    fontsize='medium'  )
plt.figure(figsize=(15,6))

sns.set_color_codes("pastel")

pl1 = sns.barplot(x="last_update", y=df_dates.sort_index().Recovered.cumsum(), data=df_dates,

            label="Total", color="g").set_title("Cumulative Recovered cases of Covid-19")

plt.xticks(rotation=90,horizontalalignment='right',

    fontweight='light',

    fontsize='medium'  )
plt.figure(figsize=(15, 6))

sns.set_color_codes("pastel")

pl1 = sns.barplot(x="last_update", y=df_dates.sort_index().Death.cumsum(), data=df_dates,

            label="Total", color="r").set_title("Cumulative Deaths cases of Covid-19")

plt.xticks(rotation=90,horizontalalignment='right',

    fontweight='light',

    fontsize='medium'  )
plt.figure(figsize=(15, 6))

X = np.arange(1)

plt.bar(X + 0.00, df_dates.Confirmed.sum(), color = 'yellow', width = 0.25, label="Confirmed")

plt.bar(X + 0.25, df_dates.Death.sum(), color = 'red', width = 0.25, label = "Death")

plt.title("Confirmed vs Deaths")

plt.grid(True)

plt.legend()

plt.show()
plt.figure(figsize=(15, 6))

X = np.arange(1)

plt.bar(X + 0.00, df_dates.Recovered.sum(), color = 'g', width = 0.25, label = "Recovered")

plt.bar(X + 0.25, df_dates.Death.sum(), color = 'r', width = 0.25, label = "Death")

plt.title("Deaths vs Recovered Cases")

plt.grid(True)

plt.legend()

plt.show()
plt.figure(figsize=(15, 6))

df_covid.Country.value_counts().plot("bar")

plt.grid(True)
plt.figure(figsize=(15, 6))

df_covid[df_covid.Country == "United States"]["State"].value_counts().plot("bar")

plt.grid(True)
df_train = df_covid.groupby("last_update")[["last_update","Confirmed"]].mean().reset_index()

df_train = df_train.set_index('last_update')

df_train.index
df_train.head(2)
#y = df_train['Confirmed']

df_train.plot(figsize=(15, 6))

plt.show()
# Define the p, d and q parameters to take any value between 0 and 2

p = d = q = range(0, 2)



# Generate all different combinations of p, q and q triplets

pdq = list(itertools.product(p, d, q))



# Fit a Model for each param value

for param in pdq:

    try:

        #print(param)

        model = ARIMA(df_train,order=param )

        results = model.fit(disp=0)

        print('ARIMA - AIC:',(param, results.aic))

    except:

        continue

ts_model = ARIMA(df_train,order=(0, 1, 0))

ts_results = ts_model.fit(disp=1)

print(ts_results.summary())
# Plot residual errors

residuals = pd.DataFrame(ts_results.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0], figsize=(10,5))

residuals.plot(kind='kde', title='Density', ax=ax[1],figsize=(10,5))

plt.show()
# Actual vs Fitted

ts_results.plot_predict(dynamic = False)

plt.show()
pred = pd.DataFrame(ts_results.predict(start = '2020-01-22', end = '2020-02-05', typ = 'levels'))

pred = pred.reset_index()

pred.columns = ['Date','Prediction']

prediction = pred.Prediction

prediction
df_train = df_train.reset_index()

actual = df_train.Confirmed

actual
mape = np.mean(np.abs(prediction - actual)/np.abs(actual))  # MAPE

rmse = np.mean((prediction - actual)**2)**.5  # RMSE

print(mape,rmse)