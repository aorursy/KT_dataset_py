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
from pandas_datareader import data as web
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
%matplotlib inline


def highlight_max_yellow(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_max(data, color='yellow'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
    
def highlight_max_all(s):
    is_max = s == s.max()
    return ['background-color: #b5f5d4' if v else '' for v in is_max]


def highlight_greaterthan(s,column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= 1
    return ['background-color: red' if is_max.any() else '' for v in is_max]
 

    
def highlight_min(data):
    color_min= '#f59d71' #green   
    attr = 'background-color: {}'.format(color_min)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else: 
        is_min = data.groupby(level=0).transform('min') == data
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)
    
df_1929 = pd.read_csv('../input/sp-500-daily-crash-1987-vs-crash-2020/DOW_1929.csv')
df_1987 = pd.read_csv('../input/sp-500-daily-crash-1987-vs-crash-2020/SP500_1987.csv')
df_2008 = pd.read_csv('../input/sp-500-daily-crash-1987-vs-crash-2020/SP500_2008.csv')
df_2020 = pd.read_csv('../input/sp-500-daily-crash-1987-vs-crash-2020/SP500_2020.csv')

#Auto update
start_20 = '2020-2-11'
end_20 = date.today()
df_20 = web.DataReader('^GSPC','yahoo',start = start_20, end = end_20)

df_2020 = df_20.copy()

d_counter = len(df_2020)

df_1929.columns = ['Date','Adj Close']

df_1929 = df_1929[31:]
#(df_1929[27:]['Adj Close'].pct_change(1)*100).head(30)

df_1929.head()
df_1987.head()
df_2008.head()
df_2020.tail()
d = pd.concat([df_1929.describe()['Adj Close'],df_1987.describe()['Adj Close'],df_2008.describe()['Adj Close'],df_2020.describe()['Adj Close']],axis=1)
d.columns = ['DOW_1929','SP500_1987','SP500_2008','SP500_2020']
pd.concat([d[['DOW_1929','SP500_1987','SP500_2008','SP500_2020']][1:2],d[['DOW_1929','SP500_1987','SP500_2008','SP500_2020']][3:4],d[['DOW_1929','SP500_1987','SP500_2008','SP500_2020']][7:8]])

df_2020 = df_2020.reset_index()
fig = go.Figure(data=[go.Candlestick(x=df_2020['Date'],
                open=df_2020['Open'],
                high=df_2020['High'],
                low=df_2020['Low'],
                close=df_2020['Close'])])

fig.update_layout(
    title='Crash 2020',
    yaxis_title='SP500 Index',
    xaxis_title='Date'
)

fig.show()
fig = go.Figure(data=[go.Candlestick(x=df_2008['Date'],
                open=df_2008['Open'],
                high=df_2008['High'],
                low=df_2008['Low'],
                close=df_2008['Close'],
                )])

fig.update_layout(
    title='Crash 2008',
    yaxis_title='SP500 Index',
    xaxis_title='Date'
)

fig.show()

fig = go.Figure(data=[go.Candlestick(x=df_1987['Date'],
                open=df_1987['Open'],
                high=df_1987['High'],
                low=df_1987['Low'],
                close=df_1987['Close'],                
                )])


fig.update_layout(
    title='Crash 1987',
    yaxis_title='SP500 Index',
    xaxis_title='Date'
)

fig.show()
fig = px.line(df_1929.iloc[0:100],x='Date',y='Adj Close')
fig.update_layout(
    title='Crash 1929',
    yaxis_title='DOW Index',
    xaxis_title='Date'
)

fig.show()
#Normalize data
c = pd.concat([df_2008['Adj Close'],df_2020['Adj Close']],axis=1)
c = c/c.iloc[0] * 100
c.columns = ['Adj Close 08','Adj Close 20']

sns.set_style('whitegrid')
plt.figure(figsize=(15,6))

c['Adj Close 08'].plot(label='Norm close 2008')
c['Adj Close 20'].plot(label='Norm close 2020')

#plt.title('Crash 2020 vs Crash 2008')
#plt.xlabel('Days')

plt.hlines(y=88,xmin=0,xmax=65, color='brown', alpha=0.5 , lw=2 , label = '-12%')
plt.hlines(y=66.5,xmin=0,xmax=65, color='r', alpha=0.5 , lw=5 , label = '-33.5%')
plt.hlines(y=59,xmin=0,xmax=65, color='black', alpha=0.5 , lw=5 , label = '-42%')
plt.legend()
plt.show()

#Normalize data
c = pd.concat([df_1987['Adj Close'],df_2020['Adj Close']],axis=1)
c = c/c.iloc[0] * 100
c.columns = ['Adj Close 87','Adj Close 20']

sns.set_style('whitegrid')
plt.figure(figsize=(15,6))
c['Adj Close 87'].plot(label='Norm close 1987')
c['Adj Close 20'].plot(label='Norm close 2020')
#plt.title('Crash 2020 vs Crash 1987')
#plt.xlabel('Days')
plt.hlines(y=88,xmin=0,xmax=65, color='brown', alpha=0.5 , lw=2 , label = '-12%')
plt.hlines(y=66.5,xmin=0,xmax=65, color='r', alpha=0.5 , lw=5 , label = '-33.5%')

plt.legend()
plt.show()

#Normalize data

df_1929 = df_1929.reset_index().drop('index',axis=1)
c = pd.concat([df_1929['Adj Close'],df_2020['Adj Close']],axis=1)
c = c/c.iloc[0] * 100
c.columns = ['Adj Close 29','Adj Close 20']

sns.set_style('whitegrid')
plt.figure(figsize=(15,6))
c['Adj Close 29'].plot(label='Norm close 1929')
c['Adj Close 20'].plot(label='Norm close 2020')
#plt.title('Crash 2020 vs Crash 1929')
#plt.xlabel('Days')
plt.hlines(y=85,xmin=0,xmax=65, color='brown', alpha=0.5 , lw=2 , label = '-15%')
plt.hlines(y=58,xmin=0,xmax=65, color='r', alpha=0.5 , lw=5 , label = '-42%')
plt.xlim(0,65)
plt.ylim(30,110)
plt.legend()
plt.show()


#Max drawdown, 
c = pd.concat([df_1929['Adj Close'],df_1987['Adj Close'],df_2008['Adj Close'],df_2020['Adj Close']],axis=1)

daily_returns = c.pct_change(1)
round(daily_returns*100,2).describe()[3:4]
round(daily_returns*100,2).describe()[7:8]
daily_ret = round(daily_returns*100,2)
daily_ret = daily_ret[0:80]
daily_ret.columns = ['1929','1987','2008','2020']
 
cm = sns.light_palette("red", as_cmap=True)


#Crash starting
dfin = daily_ret.head(40).dropna().head(40).iloc[6:]
#dfin[['1929','1987','2008','2020']].style.apply(highlight_min)
dfin[['1929','1987','2008','2020']].style.background_gradient(cmap='viridis')

dfin[['1929','1987','2008','2020']].describe()[2:].style.background_gradient(cmap='viridis')
#2 months max/min

stats= daily_ret[1:62].describe().T
plt.figure(figsize=(8,6))
sns.barplot(x=stats.index.values,y=stats['min'])
splot = sns.barplot(x=stats.index.values,y=stats['max'])
splot.tick_params(labelsize=15)
#plt.ylabel('Min / Max')
#plt.xlabel('Crash')
#plt.title ('Crashes Max/Min High/Low')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), \
                   textcoords = 'offset points',size=20)
plt.show()

#2 weeks
dfin2 = daily_ret.head(40).dropna().head(40).iloc[10:]
dfin2.iloc[0:25].head(15).cumsum()[-1:].style.background_gradient(cmap='coolwarm')
#4 weeks
dfin2 = daily_ret.head(40).dropna().head(40).iloc[10:]
dfin2.iloc[0:25].head(30).cumsum()[-1:].style.background_gradient(cmap='viridis')

daily_ret.plot(figsize=(15,6))
plt.hlines(y=-15,xmin=0,xmax=65, color='black', alpha=0.5 , lw=4 )
plt.hlines(y=15,xmin=0,xmax=65, color='black', alpha=0.5 , lw=4 , label = 'from 0 to -15%/+15%')
plt.hlines(y=-12,xmin=0,xmax=65, color='grey', alpha=0.5 , lw=3 , label = 'from 0 to -12%/+12%')
plt.hlines(y=12,xmin=0,xmax=65, color='grey', alpha=0.5 , lw=3 )
plt.title = 'Daily Pct. Change'
plt.xlabel = 'Days'
plt.ylabel = 'Pct. Change'
plt.xlim(0,40)
plt.legend()
plt.show()



fig, ax = plt.subplots(figsize=(15,6))
plt.plot(daily_ret.cumsum())

ax.fill_between(x=[10,20],y1=[-55,-55],facecolor='orange', alpha=0.1)
ax.fill_between(x=[20,30],y1=[-55,-55],facecolor='red', alpha=0.1)
ax.hlines(y=-20,xmin=0,xmax=80, color='grey', alpha=0.5 , lw=3 , label = '-20%')
ax.hlines(y=-40,xmin=0,xmax=80, color='black', alpha=0.5 , lw=3 , label = '-40%')
plt.legend()
fig.show()


corr_check = c.iloc[1:24]
corr_check.columns =  ['1929','1987','2008','2020']
#corr_check.corr()
sns.heatmap(corr_check.corr(),cmap='YlGnBu',annot = True)
plt.show()
#Log returns
logr_2020 = np.log(1+ df_2020['Adj Close'].pct_change())
#mean
u = logr_2020.mean()
#variance
var = logr_2020.var()
#drift
drift = u - (0.5 * var)
#stdev
stdev = logr_2020.std()

from scipy.stats import norm

t_intervals = 30
iterations = 250

d_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))
S0 = df_2020['Adj Close'].iloc[-1]
price_list = np.zeros_like(d_returns)
price_list[0] = S0

#x=range(0,t_intervals)
#canal_h = -1.2*x 
#canal_l = -1.3*x

for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * d_returns[t]

plt.figure(figsize=(14,10))
plt.plot(price_list,label='S&P500 Forecast')
#plt.plot(range(3000,2500,-17),lw=3,color='black',label='Channel')
#plt.plot(range(2000,1500,-17),lw=3,color='black')
#plt.title('S&P 30 days forecasting')

plt.show()

df_2020['Volume']

fig, ax = plt.subplots(figsize=(15,6))
ax.set_title('Crash 2020 Volume')
ax.set_ylabel('Volume')
fig = sns.barplot(x='Date',y='Volume',data=df_2020, palette='winter')
plt.xticks(rotation=90,visible=False)
#ax2 = ax.twinx()
#color = 'tab:green'
#ax2.set_ylabel('Adj price', fontsize=16, color=color)
#ax2 = sns.lineplot(x='Date', y='Adj Close', data = df_2020, color=color)
#ax2.tick_params(axis='y', color=color)
plt.hlines(y=0.4e10,xmin=0,xmax=60,color='red',lw=3,label='Volume before crash')
plt.legend()
plt.show()

