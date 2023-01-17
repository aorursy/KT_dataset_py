%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
# daily percentage change in the stock prices

df = pd.read_csv('../input/downine/Dow_Nine.csv')

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace = True)

df = df.pct_change()*100

df.dropna(inplace = True)

df.head()
RRdf = pd.DataFrame({'Risk':[1.2,1.8, 1.8], 

                     'Return':[1.2,1.2,1.8],

                     'Stock':['A','B','C']}, index = [0,1,2])

sns.scatterplot(x = 'Risk', y = 'Return',style = 'Stock', markers = {"A":"s", "B":"X", "C":"^"}, data = RRdf, s = 400, color = 'black');
fig = plt.figure(figsize = (10,10))

i = 1

for company in df.columns:

    ax = fig.add_subplot(3, 3, i)

    ax.scatter(df[f'{company}'], df[f'{company}'].shift(), color = 'mediumpurple', alpha = 0.5)

    ax.set_title(f'{company}')

    ax.set_xlabel('t')

    ax.set_ylabel('t+1')

    ax.label_outer()

    i = i + 1
fig = plt.figure(figsize = (15, 8))

i = 1

Normal = np.random.normal(size = 2517)

for company in df.columns:

    ax = fig.add_subplot(3,3,i)

    sns.kdeplot(Normal, bw = 1.0, color = 'red', ax = ax)

    sns.kdeplot(df[f'{company}'], color = 'blue', ax = ax)

    i = i + 1
df.boxplot(figsize = (15,8), vert = False);
# Mean Return

df.mean().sort_values().plot(kind = 'bar', color = 'lime', figsize = (10, 4));
# SD

df.std().sort_values().plot(kind = 'bar', color = 'lightcoral', figsize = (10, 4));
# Skewness

df.skew().sort_values().plot(kind = 'bar', color = 'turquoise', figsize = (10, 4));
# kurtosis

df.kurtosis().sort_values().plot(kind = 'bar', color = 'orange', figsize = (10, 4));
# Sharpe Ratio

s_ratio = pd.DataFrame({'Sharpe Ratio': df.mean()/df.std()}, index = df.columns)

s_ratio.plot(kind = 'bar', color = 'red', figsize = (10, 4));
fig = plt.figure(figsize = (15,15))

date = ['2009-01-02', '2010-01-02', '2011-01-02', '2012-01-02', '2013-01-02',

        '2014-01-02', '2015-01-02', '2016-01-02', '2017-01-02', '2018-01-02']

Year = ['2009', '2010', '2011', '2012', '2013',

        '2014', '2015', '2016' , '2017']

for i in np.arange(9):

    cor = np.round(df[date[i]:date[i+1]].corr(), decimals =2)

    m = np.min(cor.min())

    ax = fig.add_subplot(3, 3, i+1)

    sns.heatmap(cor, vmin = m, linewidths = 0.9, annot= True,  cmap='YlGnBu' , cbar = False, ax = ax)

    ax.set_title(f'{Year[i]}')
# Equal weighted portfolio

weighted_portfolio = np.zeros((9,9))

for i in np.arange(9):

    weighted_portfolio[i,0:i+1] = np.repeat(1/(i+1), i+1)

weighted_portfolio = np.round(weighted_portfolio, decimals = 3)

weighted_portfolio
w = np.asmatrix(weighted_portfolio)

c = np.asmatrix(df.cov())

SD = np.zeros(9)

for i in np.arange(9): 

    SD[i] = np.sqrt(w[i] * c * (w[i].T))

Diversification = pd.DataFrame({'SD': SD}, index = np.arange(1,10))

Diversification.plot(color = 'black', marker = 'o', markerfacecolor = 'red', 

                     markersize = 15 , figsize = (12,5), legend = False)

plt.xlabel('Number of Stocks', fontsize = 15)

plt.ylabel('Standard Deviation',fontsize = 15);
w = weighted_portfolio[8]

p = np.asmatrix(df)

a = w * p.T

df['Portfolio'] = a.T

fig = plt.figure(figsize = (25, 12))

i = 1

for company in df.columns[0:9]:

    ax = fig.add_subplot(3, 3, i)

    ax.plot(df[f'{company}'], color = 'cyan')

    ax.plot(df['Portfolio'], color = 'black')

    ax.set_title(f'{company}')

    plt.xticks(rotation=10)

    i = i+1
df.drop(columns = ['Portfolio'], inplace = True)

Mean = np.zeros(5000)

SD = np.zeros(5000)

Sharpe = np.zeros(5000)

for i in np.arange(5000):

    k = np.random.normal(size = (df.shape[1]))

    k = np.asmatrix(k/sum(k))

    p = np.asmatrix(df.mean())

    c = np.asmatrix(df.cov())

    mu = k[0] * p.T

    sigma = np.sqrt(k[0] * c * (k[0].T))

    if sigma[0, 0] > 2 :

        ()

    else:

        Mean[i] = mu

        SD[i] = sigma[0,0]

        Sharpe[i] = mu/sigma[0,0]

Simulation_df = pd.DataFrame({'Mean': Mean, 'SD': SD, 'Sharpe':Sharpe}, index = np.arange(5000))

Simulation_df = Simulation_df.replace(0, np.NaN).dropna(axis = 0)

print(Simulation_df.shape)
fig = plt.figure(figsize = (5,6))

plt.scatter(Simulation_df['SD'], Simulation_df['Mean'],c = Simulation_df['Sharpe'], cmap = 'spring_r', alpha = 0.2)

plt.xlabel('Standard Deviation')

plt.ylabel('Mean')

plt.colorbar();
fig = plt.figure(figsize = (15,8))

ax = fig.add_subplot(111, projection = '3d')

ax.scatter(Simulation_df['SD'], Simulation_df['Mean'], Simulation_df['Sharpe'], c = Simulation_df['Sharpe'], cmap = 'YlGnBu', marker = 'o')

ax.set_xlabel('SD')

ax.set_ylabel('Mean')

ax.set_zlabel('Sharpe');