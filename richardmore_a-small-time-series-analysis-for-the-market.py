import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model as lm

import kagglegym



%matplotlib inline

# Create environment

env = kagglegym.make()



# Get first observation

observation = env.reset()



# Get the train dataframe

train = observation.train
df = train
rtn_m = df.groupby('timestamp')["y"].mean() # the market return 

vol_m =  df.groupby('timestamp')["y"].std() # the market return volatility cross section

sharp_m = rtn_m/vol_m # sharp ratio

num_m = df.groupby('timestamp')["y"].count() # support number 
sns.tsplot(rtn_m)
import statsmodels.api as sm

from statsmodels.graphics.api import qqplot
def make_corelation(dta,lags):

    fig = plt.figure(figsize=(12,8))

    ax1=fig.add_subplot(211)

    fig = sm.graphics.tsa.plot_acf(dta,lags=lags,ax=ax1)

    ax2 = fig.add_subplot(212)

    fig = sm.graphics.tsa.plot_pacf(dta,lags=lags,ax=ax2)
make_corelation(rtn_m.values,20)
sns.distplot(rtn_m)
sns.tsplot(vol_m)
make_corelation(vol_m.values,20)
sns.distplot(vol_m)
sns.tsplot(sharp_m)
make_corelation(sharp_m.values,20)
sns.distplot(sharp_m)
sns.tsplot(num_m)
