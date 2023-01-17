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
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
%matplotlib inline 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-21-2020.csv')
ts_us_confirmed = pd.read_csv('/kaggle/input/covidak/time_series_covid19_confirmed_US.csv')
ts_us_deaths = pd.read_csv('/kaggle/input/covidak/time_series_covid19_deaths_US.csv')
ts_global_confirmed = pd.read_csv('/kaggle/input/covid-19-cssegisanddata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
ts_global_deaths = pd.read_csv('/kaggle/input/covid-19-cssegisanddata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
ts_us_confirmed.head()
ts_us_deaths.head()
ts_global_confirmed.head()
ts_global_deaths.head()
# Find top deaths

def findDateStart(keys):
    i =0
    for k in keys:
        i += 1
        if k == 'Population':
            break;
    return i

def plot_ts_values(df, number, func=None, title=None):
    ts_us_deaths_sorted_raw = df.sort_values(by=['4/22/20'], ascending=False)
    ts_us_deaths_sorted = ts_us_deaths_sorted_raw.copy()
    ts_us_deaths_sorted['key'] = ts_us_deaths_sorted['Admin2']  + ':' + ts_us_deaths_sorted['Province_State'] 
    ts_us_deaths_sorted = ts_us_deaths_sorted.set_index('key')
    cols = ts_us_deaths_sorted.keys()
    ts_us_deaths_sorted = ts_us_deaths_sorted.loc[:,cols[findDateStart(cols)]:cols[-1]]
    if (func != None):
        ts_us_deaths_sorted = ts_us_deaths_sorted.apply(func)
    ts_us_deaths_sorted_100 = ts_us_deaths_sorted.head(number)
    ts_us_deaths_sorted_100 = ts_us_deaths_sorted_100.T
    plt.figure(figsize=(16, 9))
    cols = ts_us_deaths_sorted_100.keys()
    days = list(range(ts_us_deaths_sorted_100.index.size))
    dfc = ts_us_deaths_sorted_100.replace([np.inf, -np.inf, np.NaN], 0)
    for c in cols:
        #plt.plot(ts_us_deaths_sorted_100.index, ts_us_deaths_sorted_100[c])
        plt.plot(days[40:-1], dfc[c][40:-1])
    if (title == None):
        title = 'Top # of Cases'
    plt.title(title, size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(cols, prop={'size': 10})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
    #print(ts_us_deaths_sorted_100)
plot_ts_values(ts_us_deaths, 20, title = 'Top # of Cases by Area(log)')
plot_ts_values(ts_us_deaths, 20, np.log, title = 'Top # of Cases by Area(log)')
def adjustForPopulation(df):
    dfc = df.copy()
    cols = dfc.keys()
    k = findDateStart(cols)
    for c in cols[k:-1]:
        dfc[c] = dfc[c] * 1000000 / dfc['Population']
    dfc = dfc.replace([np.inf, -np.inf], 0)
    return dfc
ts_us_deaths_adj = adjustForPopulation(ts_us_deaths)
plot_ts_values(ts_us_deaths_adj, 20, title='# of Cases by Area/1M')
plot_ts_values(ts_us_deaths_adj, 20, np.log, title='# of Cases by Area/1M (log)')
ts_us_deaths_bigger = ts_us_deaths[ts_us_deaths['Population'] > 200000]
plot_ts_values(ts_us_deaths_bigger, 20, title='# of Cases by Area (pop>200K)')
plot_ts_values(ts_us_deaths_bigger, 20, np.log, title='# of Cases by Area (pop>200K) (log)')

ts_us_deaths_bigger_adj = adjustForPopulation(ts_us_deaths_bigger)
plot_ts_values(ts_us_deaths_bigger_adj, 20, title='# of Cases/1M (pop>200K)')
plot_ts_values(ts_us_deaths_bigger_adj, 20, np.log, title='# of Cases/1M -log (pop>200K)')
# get dataframe sorted by life Expectancy in each continent
def findTopByState(df, number):
    g = df.groupby(["Province_State"]).apply(lambda x: x.sort_values(["4/22/20"], ascending = False)).reset_index(drop=True)
    # select top N rows within each continent
    g=g.groupby('Province_State').head(number)
    return g
ts_us_deaths_top = findTopByState(ts_us_deaths, 2)
plot_ts_values(ts_us_deaths_top, 30, title = 'Top 2 # of Cases by State(log)')
plot_ts_values(ts_us_deaths_top, 30, np.log, title = 'Top 2 # of Cases by State(log)')
ts_us_deaths_top_bigger_adj = findTopByState(ts_us_deaths_bigger_adj, 2)
plot_ts_values(ts_us_deaths_top_bigger_adj, 20, title = 'Top 2 # of Cases by State/1M')
plot_ts_values(ts_us_deaths_top_bigger_adj, 20, np.log, title = 'Top 2 # of State/1M(log)')
