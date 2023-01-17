import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import FunctionTransformer

import matplotlib.pyplot as plt

import warnings  

warnings.filterwarnings('ignore')

pd.set_option("display.precision", 3)

pd.set_option("display.expand_frame_repr", False)

pd.set_option("display.max_rows", 25)

#covid_19_data = pd.read_csv("../input/data/time-series-19-covid-combined.csv

covid_19_data = pd.read_csv("../input/data/time-series-19-covid-combined.csv")

by_date = covid_19_data.groupby(['Date','Country/Region'])[['Confirmed']].agg("sum")

country_cols = by_date['Confirmed'].unstack()

countries = set(list(covid_19_data['Country/Region']))

#print(countries)

countries = ['Australia','US','Italy','Korea, South','United Kingdom']

doubling_time = {}

doubling_ts = {}

for COL in countries:

    a = country_cols[COL].copy()[:]

    a_reset = a.reset_index()

    ydf = a_reset[COL].fillna(value=0)

    x = ydf.index.values

    y = ydf.values

    ygrad = np.gradient(np.log(y))

    ygrad[np.isnan(ygrad)] = 0.000

    #ygrad[np.isinf(ygrad)] = 0.0001

    ydouble = np.log(2)/ygrad

    # Linearised:

    # y = ab^x = ae^{lnb * x}

    # y = B * exp{A}^{x}

    # for y = x_o * b^{t}

    # Tdouble = log(2)/log(b) - in this one Tdoube = log(2)/log(exp(A))

    doubling_ts[COL] = ydouble

    

DFD = pd.DataFrame(doubling_ts)

DFD = DFD.rolling(3, win_type='gaussian').mean(std=2)

DFD['date'] = covid_19_data['Date']

DFD = DFD.set_index('date')

CFD = country_cols[countries]

CDFD = pd.concat([DFD,CFD],axis=1,keys=['Doubling','Cases']).swaplevel(0,1,axis=1).sort_index(axis=1)
CDFD.loc['2020-03-14':]
#DFD[['Australia','US','Italy','United Kingdom']].loc['2020-03-10'::3].plot(kind='bar',figsize=(12, 6))

DFD[['Australia','US','Italy','United Kingdom']].loc['2020-03-10':].plot(kind='line',style='o-',figsize=(12, 6))

plt.title('Doubling rates: 7 March - 25 March. (Higher is better!)')
CFD.loc['2020-03-07':'2020-03-25'].plot(figsize=(12, 6))

fig = plt.gcf()

fig.autofmt_xdate()

plt.title('Cases 7 March until 25 March')