

import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

from scipy.stats import norm

%matplotlib inline
df = pd.read_csv('/kaggle/input/earthquakes-near-istanbul-for-last-1-years/dataist.csv')

df
df.describe()
df.columns
df.info()
df['Times'] = pd.to_datetime(df[' Time '], format='%Y%m%d', errors='ignore')
plt.figure(figsize=(12.5,4.5))

plt.plot(df.index, df[' Magnitude '], color='red')

plt.xlabel('Number of Earthquake',color='cyan')

plt.ylabel('Earthquake Intensity',color='cyan')

plt.grid(True)

plt.title("Earthquakes Near Istanbul", color='cyan')

plt.show()


plt.scatter(x=df[' Time '], y=df[' Magnitude '])

plt.xticks(rotation=120)

plt.xlabel('Date of Earthquakes',color='cyan')

plt.ylabel('Earthquake Intensity',color='cyan')



plt.title("Earthquakes Near Istanbul", color='cyan')

plt.show()

#Last 1 year eartquakes' average

df[' Magnitude '].mean()
#Monte Carlo Simulation for future Earthquakes

ticker = '/kaggle/input/earthquakes-near-istanbul-for-last-1-years/dataist.csv'

data = pd.DataFrame()

data[ticker] = pd.read_csv('/kaggle/input/earthquakes-near-istanbul-for-last-1-years/dataist.csv')[' Magnitude ']

log_returns= np.log(1 + data.pct_change())

log_returns.tail()



log_returns.plot(figsize=(10,6))

plt.show()
u = log_returns.mean()

u
var = log_returns.var()

var




drift = u- (0.5 * var)

drift



stdev = log_returns.std()

stdev
type(drift)
type(stdev)
np.array(drift)
drift.values

stdev.values




norm.ppf(0.95)



x = np.random.rand(10,2)

x




norm.ppf(x)



Z = norm.ppf(np.random.rand(10,2))

Z
t_intervals = 365

iterations = 5
daily_returns = np.exp(drift.values + stdev.values*norm.ppf(np.random.rand(t_intervals, iterations)))
daily_returns
S0 = data.iloc[-1]

S0




price_list = np.zeros_like(daily_returns)

price_list
price_list[0] = S0

price_list
for t in range(1, t_intervals):

  price_list[t] = price_list[t-1] * daily_returns[t]

price_list
plt.figure(figsize=(10,6))

plt.plot(price_list) 

plt.grid(True)

plt.xlabel('DAYS', color='orange')

plt.ylabel('Magnitude', color='orange')

plt.title('Monte Carlo Simulation for Istanbul Earthquake',color='cyan')

plt.show()