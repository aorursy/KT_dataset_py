#Importing Libraries

import pandas as pd

import numpy as np

import seaborn as sns

import scipy.stats

import matplotlib.pyplot as plt

import plotly.express as px

! pip install chart_studio

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

%matplotlib inline
#Reading data for Nifty50, Nifty Midcap and Nifty Smallcap



nifty_50 = pd.read_csv('../input/nifty-indices-dataset/NIFTY 50.csv',parse_dates=['Date'])

nifty_midCap = pd.read_csv('../input/nifty-indices-dataset/NIFTY MIDCAP 150.csv',parse_dates=['Date'])

nifty_smallCap = pd.read_csv('../input/nifty-indices-dataset/NIFTY SMALLCAP 250.csv',parse_dates=['Date'])
def minMaxDate(df):

    return (df['Date'].min(),df['Date'].max())
print(minMaxDate(nifty_50))

print(minMaxDate(nifty_midCap))

print(minMaxDate(nifty_smallCap))

## All these datasets have common dates from 1st April 2005
# Merging only the closing prices

df = nifty_midCap[['Date','Close']].merge(nifty_smallCap[['Date','Close']],on='Date',how='left')

df = df.merge(nifty_50[['Date','Close']], on = 'Date', how='left')

df.columns = ['Date','NiftyMidCap','NiftySmallCap','Nifty50']
# Setting Date columns as index, aggregating closing prices at month level

df = df.set_index('Date').resample('M').last()

df.index = df.index.to_period("M")

df.head(4)
# Dataframe with calculated percent change between two consecutive index price closing

df_rets = df.pct_change().dropna()

df_rets.head(4)
# Dataframe showing how much INR 100 would be with time. 

# This will exactly track the indices and values are comparable for indices

df_wealth = 100 * (1+df_rets).cumprod()

df_wealth.head(4)
df_rets.corr().style.background_gradient(cmap='Greens',axis=0,text_color_threshold=0.4)

## Correlation of returns for indices show that there is very high (>0.85) correlation among indices
# Line plot of index prices

df.iplot(title='Line plot')



#12 months moving average of index prices

df.rolling(12).mean().iplot(title='12 months moving average')
df_rets.iplot(title='Line plot of returns')
## Functions to create a summary of risk and return characteristics 



def skewness(r):

    """

    Calculates skewness

    """

    return scipy.stats.skew(r)



def kurtosis(r):

    """

    Calculates kurtosis

    """

    return scipy.stats.kurtosis(r)



def returns(r, periods_per_year):

    """

    Calculates annualized returns

    """

    compounded_growth = (1+r).prod()

    n_periods = r.shape[0]

    return compounded_growth**(periods_per_year/n_periods)-1



def volatility(r, periods_per_year):

    """

    Calculates annualized volatility of returns

    """

    return r.std()*(periods_per_year**0.5)



def sharpe_ratio(r, rfr, periods_per_year):

    """

    Calculates annualized sharpe ratio of a set of returns

    """

    rf_per_period = (1+ rfr)**(1/periods_per_year)-1

    excess_ret = r - rf_per_period

    ann_ex_ret = returns(excess_ret, periods_per_year)

    ann_vol = volatility(r,periods_per_year)

    return ann_ex_ret/ann_vol



def summary_stats(r, rfr = 0.05):

    """

    Returns a Dataframe that contains aggregated summary stats for the returns in columns of r

    """

    ann_r = r.aggregate(returns, periods_per_year = 12)

    ann_vol = r.aggregate(volatility, periods_per_year = 12)

    ann_sr = r.aggregate(sharpe_ratio,rfr=rfr,periods_per_year=12)

    skew = r.aggregate(skewness)

    kurt = r.aggregate(kurtosis)

    

    return pd.DataFrame({

        'Annualized Return': ann_r,

        'Annualized Vol': ann_vol,

        'Skewness':skew,

        'Kurtosis':kurt,

        'Sharpe Ratio': ann_sr,

    })
## Overall Summary stats for before 2018 periods

summary_stats(df_rets).style.background_gradient(cmap='Greens',axis=0,text_color_threshold=0.4)
## Summary stats for before 2018 periods

summary_stats(df_rets[:'2018']).style.background_gradient(cmap='Greens',axis=0,text_color_threshold=0.4)
## Summary stats for after 2018 periods

summary_stats(df_rets['2018':]).style.background_gradient(cmap='Greens',axis=0,text_color_threshold=0.4)
returns(df_rets[:'2018'],12).iplot(kind='bar',color='green',opacity=0.4)
returns(df_rets['2018':],12).iplot(kind='bar',color='red',opacity=0.4)
volatility(df_rets,12).iplot(kind='bar',color='blue')
risk_free_rate = 0.05

sharpe_ratio(df_rets,risk_free_rate,12).iplot(kind='bar',color='black',opacity=0.4)
df_rets.iplot(kind='hist',histnorm='percent')
df_wealth.iplot(asFigure=True,

               vspan={'x0':'2017-11-01','x1':'2020-05-03',

                      'color':'rgba(30,30,30,0.3)','color':'black','fill':True,'opacity':.2})
df_wealth['2018':].corr().style.background_gradient(cmap='Greens',axis=0,text_color_threshold=0.4)
df_wealth['Nifty50'].rolling(6).corr(df_wealth['NiftySmallCap']).iplot()