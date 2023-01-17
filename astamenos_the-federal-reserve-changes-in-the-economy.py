# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
sns.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def loadDataframe(filename):
    df = pd.read_csv(filename, index_col = 'DATE', parse_dates = True)
    print(df.info())
    
    return df

def tidyDataframe(df, col_names):
    df.columns = col_names
    print(df.info())
    
    return df

def fedChair(df):
    df.loc[:'1918-12-15', 'chair'] = 'McAdoo'
    df.loc['1918-12-16':'1920-02-01', 'chair'] = 'Glass'
    df.loc['1920-02-02':'1921-03-03', 'chair'] = 'Houston'
    df.loc['1921-03-04':'1932-02-12', 'chair'] = 'Mellon'
    df.loc['1932-02-13':'1933-03-04', 'chair'] = 'Mills'
    df.loc['1933-03-05':'1933-12-31', 'chair'] = 'Woodin'
    df.loc['1934-01-01':'1936-02-01', 'chair'] = 'Morgenthau'
    df.loc['1936-02-02':'1948-01-31', 'chair'] = 'Eccles'
    df.loc['1948-02-01':'1951-03-31', 'chair'] = 'McCabe'
    df.loc['1951-04-01':'1970-01-31', 'chair'] = 'Martin'
    df.loc['1970-02-01':'1978-01-31', 'chair'] = 'Burns'
    df.loc['1978-03-08':'1979-08-06', 'chair'] = 'Miller'
    df.loc['1979-08-07':'1987-08-11', 'chair'] = 'Volcker'
    df.loc['1987-08-12':'2006-01-31', 'chair'] = 'Greenspan'
    df.loc['2006-02-01':'2014-02-02', 'chair'] = 'Bernanke'
    df.loc['2014-02-03':'2018-02-03', 'chair'] = 'Yellen'
    df.loc['2018-02-04':, 'chair'] = 'Powell'
    
    return df

def decomposition(df, column, f):
    decomp = seasonal_decompose(df[column], model = 'additive', freq = f)
    return decomp.observed, decomp.trend, decomp.seasonal, decomp.resid

def constructPlots(df, decomp1, decomp2, decomp3, decomp4, title, a, c):
    sns.color_palette('dark')
    plt.clf()
    fig, axes = plt.subplots(nrows = 4, figsize = (25, 25))
    
    sns.lineplot(df.index, y = decomp1, hue = 'chair', data = df, ax = axes[0])
    sns.lineplot(df.index, y = decomp2, hue = 'chair', data = df, ax = axes[1])
    sns.lineplot(df.index, y = decomp3, hue = 'chair', data = df, ax = axes[2])
    sns.lineplot(df.index, y = decomp4, hue = 'chair', data = df, ax = axes[3])
    
    axes[0].set(title = title, xlabel = 'Time', ylabel = 'Observed Change (in %)')
    axes[1].set(xlabel = 'Time', ylabel = 'Trend')
    axes[2].set(xlabel = 'Time', ylabel = 'Seasonal')
    axes[3].set(xlabel = 'Time', ylabel = 'Residual')
    
    axes[0].axhline(y = 0)
    axes[1].axhline(y = 0)
    axes[2].axhline(y = 0)
    axes[3].axhline(y = 0)
    
    recessions = [df['1913-01':'1914-12'], df['1918-08':'1919-03'], df['1920-01':'1921-07'], 
                  df['1923-05':'1924-07'], df['1926-10':'1927-11'], df['1929-08':'1933-03'], 
                  df['1937-05':'1938-06'], df['1945-02':'1945-10'], df['1948-11':'1949-10'], 
                  df['1953-07':'1954-05'], df['1957-08':'1958-04'], df['1960-04':'1961-02'],
                  df['1969-12':'1970-11'], df['1973-11':'1975-03'], df['1980-01':'1980-07'], 
                  df['1981-07':'1982-11'], df['1990-07':'1991-03'], df['2001-03':'2001-11'], 
                  df['2007-12':'2009-06'], df['2020-02':]]
                    
    # plots recessions for each graph
    for i in range(4):
        for r in recessions:
            if r.shape[0] != 0:
                axes[i].axvspan(r.index[0], r.index[-1], color = c, alpha = a)

def constructDistrPlots(df, column, label, title1, title2):
    fig, axes = plt.subplots(nrows = 2, figsize = (18, 18))
    sns.distplot(df[column], kde = True, ax = axes[0])
    sns.boxplot(x = 'chair', y = column, data = df, ax = axes[1])
    axes[0].set(xlabel = label, ylabel = 'Frequency', title = title1)
    axes[1].set(xlabel = 'Federal Reserve Chair', ylabel = label, title = title2)

def stationarityTest(df, column, k):
    roll_avg = df[column].rolling(k).mean()
    roll_std = df[column].rolling(k).std()
    
    fig, ax = plt.subplots(figsize = (18, 18))
    sns.lineplot(x = df.index, y = roll_avg, color = 'steelblue', ax = ax, label = 'Rolling Average')
    sns.lineplot(x = df.index, y = roll_std, color = 'firebrick', ax = ax, label = 'Rolling Standard Deviation')
    
    test = adfuller(df[column], autolag = 'AIC')
    print('Results of Dickey-Fuller Test')
    print('Test Statistic: ', test[0])
    if test[1] < 0.001:
        print('p-value: <0. 001')
    else:
        print('p-value: ', test[1])
    print('Number of Lags Used: ', test[2])
    print('Number of Observations: ', test[3])
    print('Critical Values: ', test[4])
gdp_filename = '/kaggle/input/percent-change-in-gdp-from-preceding-period/A191RL1Q225SBEA.csv'
gdp_col_names = ['gdp_change']
gdp_title = '% Change in Real GDP from Preceding Period'
gdp = loadDataframe(gdp_filename)
gdp = tidyDataframe(gdp, gdp_col_names)
gdp = fedChair(gdp)
gdp_observed, gdp_trend, gdp_seasonal, gdp_resid = decomposition(gdp, 'gdp_change',f = 4)
constructPlots(gdp, gdp_observed, gdp_trend, gdp_seasonal, gdp_resid, title = gdp_title, c = 'slategray', a = 0.30)
constructDistrPlots(gdp, 'gdp_change', '% Change in Real GDP', 'Distribution of Real GDP Change', 'Real GDP Change by Chair Term')
stationarityTest(gdp, 'gdp_change', k = 4)
cpi_filename = '/kaggle/input/cpi-percent-change-19821984-100/CPIAUCNS.csv'
cpi_col_names = ['cpi_change']
cpi_title = '% Change in CPI'
cpi = loadDataframe(cpi_filename)
cpi = tidyDataframe(cpi, cpi_col_names)
cpi = cpi.resample('Q').mean()
cpi = fedChair(cpi)
cpi_observed, cpi_trend, cpi_seasonal, cpi_resid = decomposition(cpi, 'cpi_change', f = 4)
constructPlots(cpi, cpi_observed, cpi_trend, cpi_seasonal, cpi_resid, title = cpi_title, c = 'slategray', a = 0.30)
constructDistrPlots(cpi, 'cpi_change', '% Change in CPI', 'Distribution of CPI Change', 'CPI Change by Chair Term')
stationarityTest(cpi, 'cpi_change', k = 4)
cpi_transform = cpi.copy()
cpi_transform['cpi_change'] = cpi_resid
cpi_transform.dropna(inplace = True)
constructDistrPlots(cpi_transform, 'cpi_change', '% Change in Residual CPI', 'Distribution of Residual CPI Change', 'Residual CPI Change by Chair Term')
stationarityTest(cpi_transform, 'cpi_change', k = 4)