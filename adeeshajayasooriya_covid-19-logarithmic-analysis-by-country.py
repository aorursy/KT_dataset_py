#Change the country name and go all the way down to see the predictions for today

COUNTRY = "Sri Lanka"
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator 

from datetime import datetime

plt.style.use('seaborn')

%matplotlib inline 
confirmed_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

death_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

recovered_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

confirmed_df.head()
confirmed_df_t = confirmed_df.melt(

    id_vars = ["Country/Region", "Province/State","Lat","Long"],

    var_name = "Date",

    value_name="Value"

)

confirmed_df_t.head()
if(COUNTRY=="World"):

    confirmed_df_t_sl = confirmed_df_t.drop(["Country/Region","Province/State","Lat","Long"], axis=1)

    confirmed_df_t_sl = confirmed_df_t_sl.groupby(["Date"]).sum()





else:

    confirmed_df_t_sl = confirmed_df_t[confirmed_df_t['Country/Region']==COUNTRY]

    confirmed_df_t_sl = confirmed_df_t_sl.drop(["Province/State","Lat","Long"], axis=1)

    confirmed_df_t_sl = confirmed_df_t_sl.groupby(["Country/Region","Date"]).sum()



confirmed_df_t_sl = confirmed_df_t_sl.reset_index()

confirmed_df_t_sl["Date"] = pd.to_datetime(confirmed_df_t_sl['Date'])

confirmed_df_t_sl = confirmed_df_t_sl.sort_values(by=["Date"])

confirmed_df_t_sl.tail()
confirmed_df_t_sl['Frequency'] = confirmed_df_t_sl[['Value']].diff().fillna(confirmed_df_t_sl)

confirmed_df_t_sl.head()
#USA choosed as benchmark 

confirmed_df_t_cn = confirmed_df_t[confirmed_df_t['Country/Region']=="US"]

confirmed_df_t_cn = confirmed_df_t_cn.drop(["Province/State","Lat","Long"], axis=1)



confirmed_df_t_cn = confirmed_df_t_cn.groupby(["Country/Region","Date"]).sum()

confirmed_df_t_cn = confirmed_df_t_cn.reset_index()

confirmed_df_t_cn["Date"] = pd.to_datetime(confirmed_df_t_cn['Date'])

confirmed_df_t_cn = confirmed_df_t_cn.sort_values(by=["Date"])

confirmed_df_t_cn['Frequency'] = confirmed_df_t_cn[['Value']].diff().fillna(confirmed_df_t_cn)

confirmed_df_t_cn.tail()
#SRI LANKA

confirmed_df_t_sl["Date"] = pd.to_datetime(confirmed_df_t_sl['Date'])

confirmed_df_t_sl_ts = confirmed_df_t_sl.iloc[:,-3:]

confirmed_df_t_sl_ts = confirmed_df_t_sl_ts.set_index('Date')



#USA

confirmed_df_t_cn["Date"] = pd.to_datetime(confirmed_df_t_cn['Date'])

confirmed_df_t_cn_ts = confirmed_df_t_cn.iloc[:,-3:]

confirmed_df_t_cn_ts = confirmed_df_t_cn_ts.set_index('Date')

plt.figure(figsize=(20,10))



#plt.plot(confirmed_df_t_cn_ts['Frequency'].rolling(7).sum(), label="USA - Benchmark", color="blue")

plt.plot(confirmed_df_t_sl_ts['Frequency'].rolling(7).mean(), label="actual", color="black")

plt.plot(confirmed_df_t_sl_ts['Frequency'].rolling(20).mean(), label="ma20", color="yellow")

plt.plot(confirmed_df_t_sl_ts['Frequency'].rolling(50).mean(), label="ma50", color="blue")

plt.plot(confirmed_df_t_sl_ts['Frequency'].rolling(200).mean(), label="ma200", color="red")

plt.xlabel('Total confirmed cases')

plt.ylabel('daily confirmed cases') 

plt.legend(loc="best")

plt.show()
#SRI LANKA

confirmed_df_t_sl["Date"] = pd.to_datetime(confirmed_df_t_sl['Date'])

confirmed_df_t_sl_ts = confirmed_df_t_sl.iloc[:,-3:]

confirmed_df_t_sl_ts = confirmed_df_t_sl_ts.set_index('Date')



#USA

confirmed_df_t_cn["Date"] = pd.to_datetime(confirmed_df_t_cn['Date'])

confirmed_df_t_cn_ts = confirmed_df_t_cn.iloc[:,-3:]

confirmed_df_t_cn_ts = confirmed_df_t_cn_ts.set_index('Date')

plt.figure(figsize=(20,10))



plt.plot(confirmed_df_t_cn_ts['Value'].rolling(7).mean(), confirmed_df_t_cn_ts['Frequency'].rolling(7).mean(), label="USA - Benchmark", color="blue")

plt.plot(confirmed_df_t_sl_ts['Value'].rolling(7).mean(), confirmed_df_t_sl_ts['Frequency'].rolling(7).mean(), label=COUNTRY, color="red")

plt.xlabel('Total confirmed cases')

plt.ylabel('daily confirmed cases') 

plt.legend(loc="best")

plt.xscale("log")

plt.yscale("log")

plt.show()