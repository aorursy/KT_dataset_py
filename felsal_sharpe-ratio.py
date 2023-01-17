import sys

import subprocess



REPO_LOCATION = 'https://github.com/felipessalvatore/vol4life'

REPO_NAME = 'vol4life'

REPO_BRANCH = 'master'



# Clone the repository

print('cloning the repository')

subprocess.call(['git', 'clone', '-b', REPO_BRANCH, REPO_LOCATION])



# Setting env variables

sys.path.append(REPO_NAME)

import pandas as pd

import numpy as np

import statsmodels.api as sm

from scipy.stats import norm

from IPython.display import display, HTML

from vol4life.vol4life.score import sharpe_ratio

from vol4life.vol4life.stats import get_ecdf

from vol4life.vol4life.plot import plot_acf

from vol4life.vol4life.stats import autocorrelation_f



import matplotlib.pyplot as plt

from glob import glob



path_ibov =  glob('/kaggle/input/ibovespa-stocks/b3*.csv')[0]

path_selic = glob('/kaggle/input/ibovespa-stocks/selic.csv')[0]
selic = pd.read_csv(path_selic)

selic.loc[:,"datetime"] = pd.to_datetime(selic.datetime)

selic = selic[["datetime", "selic"]].groupby("datetime").mean()
df = pd.read_csv(path_ibov)

df.loc[:, "datetime"] =  pd.to_datetime(df.datetime)



tickers = ["ITUB4", "BPAN4", "VALE3", "VVAR4"]

initial_date = "2018-07-01"

final_date = "2019-07-01"



df_sort = df.set_index(["ticker", "datetime"]).sort_index()

tss = []

for ticker in tickers:  

    ts = df_sort.xs(ticker).close

    ts.name = ticker

    tss.append(ts)



del df_sort

prices = pd.concat(tss,1).interpolate("linear", limit_direction="both")[initial_date:final_date]

returns = prices.pct_change().dropna()



# display

display(HTML("<h3>Price dataset</h3>"))

display(HTML("<br><b>head<b>"))

display(HTML(prices.head(2).to_html()))

display(HTML("<br><b>tail<b>"))

display(HTML(prices.tail(2).to_html()))

display(HTML("<br><b>shape = {}<b>".format(prices.shape)))
size = prices.shape[1]

p1 = [0,1,0,0]

return1 = prices.dot(p1).pct_change().dropna()

return1.name = "bpan4_returns"



p2 = [1,0,0,0]

return2 = prices.dot(p2).pct_change().dropna()

return2.name = "itub4_returns"



p3 = [0,0,1,0]

return3 = prices.dot(p3).pct_change().dropna()

return3.name = "vale3_returns"



p4 = [0,0,0,1]

return4 = prices.dot(p4).pct_change().dropna()

return4.name = "vvar4_returns"



p5 = [0.25,0.25,0.25,0.25]

return5 = prices.dot(p5).pct_change().dropna()

return5.name = "combined_returns"







for r in [return1, return2, return3, return4, return5]:

    r_stats = r.to_frame().describe()

    r_stats.loc['var'] = r.var()

    r_stats.loc['skew'] = r.skew()

    r_stats.loc['kurt'] = r.kurtosis()

    display(HTML(r_stats.loc[['skew', "kurt"]].to_html()))
prices
mean, var, size = return1.mean(),return1.var(), return1.shape[0]

ref = pd.Series(np.random.normal(mean,var,size))

ref.plot.hist(alpha=0.9);

return1.plot.hist(alpha=0.5);





sm.qqplot(ref,dist=norm(mean,var), line='45')

sm.qqplot(return1,dist=norm(mean,var), line='45')

plt.show()

plot_acf(return1, lag_range=10, out_path=None, acf_function=autocorrelation_f)
rfr = selic[initial_date:final_date].mean()[0]

uni_sr = sharpe_ratio(return1,rfr=rfr)



l,r = initial_date.split("-")[0], final_date.split("-")[0]



print("Average daily risk-free-rate between {} and {} = {:.5f}".format(l,r, rfr))

print("Return1 Sharpe ratio = {:.3f}".format(uni_sr))
simulations = 5000

boots_sr = []

seed = 2342



np.random.seed(seed)



for _ in range(simulations):

    boot_r = return1.sample(return1.shape[0], replace=True).reset_index(drop=True)

    boots_sr.append(sharpe_ratio(boot_r,rfr=rfr))



boots_sr = pd.Series(boots_sr)

fig, ax = plt.subplots(figsize = (10,5))

n, bins, patches = plt.hist(x=boots_sr, bins='auto', color='Green',

                            alpha=0.7, rwidth=0.85)

ax.axvline(uni_sr,0, 140, ls="--", color="k", label="observable Sharpe ratio");

msg = "Sharpe ratio distribution (bootstrap replications)\n"

ax.set_title(msg)

ax.legend(loc="best");
simulations = 5000

sr_series = []

seed = 123

size = prices.shape[1]



np.random.seed(seed)





for _ in range(simulations):

    w = np.random.uniform(0,1,size)

    w = w / np.sum(w)

    r = prices.dot(w).pct_change()

    sr = sharpe_ratio(r,rfr=rfr)

    sr_series.append(sr)

    

sr_series = pd.Series(sr_series)

sr_ecdf = get_ecdf(sr_series)



p_greater_sr = 1 - sr_ecdf(uni_sr)



fig, ax = plt.subplots(figsize = (12,6))

n, bins, patches = plt.hist(x=sr_series, bins='auto', color='#0294aa',

                            alpha=0.7, rwidth=0.85)

ax.axvline(uni_sr,0, 140, ls="--", color="k", label="uniform portfolio");

msg = "Random portfolio's Sharpe ratio distribution\n"

msg += "probability of getting a sharpe-ratio better than the uniform portfolio is {:.1%}".format(p_greater_sr)

ax.set_title(msg)

ax.legend(loc="best");
### Cleaning

print('removing the repository')

subprocess.call(['rm', '-rf', REPO_NAME])