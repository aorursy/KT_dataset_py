import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

from IPython.display import display, HTML

from itertools import combinations

from tqdm import tqdm

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import os



        

from glob import glob



path_ibov =  glob('/kaggle/input/ibovespa-stocks/b3*.csv')[0]



def pad_time_array(full_time, ts):

    pad = [i for i in full_time if i not in ts.index]

    pad_array = np.empty(len(pad))

    pad_array[:] = np.NaN

    pad_ts = pd.Series(pad_array, index=pad)

    ts_new = pd.concat([ts, pad_ts]).sort_index()

    return ts_new





def get_corr_over_time(df,ticker1, ticker2, window=100):

    ts1 = returns[ticker1]

    ts2 = returns[ticker2]

    corr = ts1.rolling(window).corr(ts2).dropna()

    corr.name = "{}_{}_corr_{}".format(ticker1, ticker2, window)

    return corr
df = pd.read_csv(path_ibov)

df.loc[:, "datetime"] = df.datetime.map(pd.Timestamp)



df_sorted = df.set_index(["ticker", "datetime"]).sort_index()
ibov = ["ABEV3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BPAC11", "BRAP4",

        "BRDT3", "BRFS3", "BRKM5", "BRML3", "BTOW3", "CCRO3", "CIEL3", "CMIG4", "COGN3", "CRFB3",

        "CSAN3", "CSNA3", "CVCB3", "CYRE3", "ECOR3", "EGIE3", "ELET3", "ELET6", "EMBR3", "ENBR3",

        "EQTL3", "FLRY3", "GGBR4", "GNDI3", "GOAU4", "GOLL4", "HAPV3", "HGTX3", "HYPE3", "IGTA3",

        "IRBR3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "LAME4", "LREN3", "MGLU3", "MRFG3", "MRVE3", "MULT3",

        "NTCO3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RAIL3", "RENT3", "SANB11", "SBSP3", "SMLS3",

        "SULA11", "SUZB3", "TAEE11", "TIMP3", "TOTS3", "UGPA3", "USIM5", "VALE3", "VIVT4", "VVAR3", "WEGE3", "YDUQ3"]





df_sort = df.set_index(["ticker", "datetime"]).sort_index()

start_date = "2007-01-01"





ibov_mini = []

for ticker in ibov:

    ts = df_sort.xs(ticker)

    if ts.index.min() <= pd.Timestamp(start_date):

        ibov_mini.append(ticker)



sizes = []



for ticker in ibov_mini:

    ts = df_sort.xs(ticker).close[start_date:]

    sizes.append(ts.shape[0])

max_size = np.max(sizes)

max_id = np.argmax(sizes)

max_ticker = ibov_mini[max_id]





full_time = df_sort.xs(max_ticker).close[start_date:].index



all_ts = []

for ticker in tqdm(ibov_mini):

    ts = df_sort.xs(ticker).close[start_date:]

    ts = pad_time_array(full_time, ts)

    ts.name = ticker

    all_ts.append(ts)

    

prices = pd.concat(all_ts,1)

prices = prices.interpolate("linear", limit_direction="both")

returns = prices.pct_change().dropna()



del df_sort



ratio = len(ibov_mini)/len(ibov)

print("percentage of ibov's tickers that will be used in the analysis = {:.1%}".format(ratio))



display(HTML(returns.head(5).to_html()))

display(HTML(returns.tail(5).to_html()))
## Selecting the top_n most correlated tickers

corr = returns.corr().abs().unstack()

corr = corr.sort_values(kind="quicksort")

corr= corr[corr < 1.0][::-1]

top_n = 4

most_ticker_corr = list(set([c[0] for c in corr[:top_n].index] + [c[1] for c in corr[:top_n].index]))

small_returns = returns[most_ticker_corr]





sns.pairplot(small_returns, diag_kind='hist')

plt.show()
corr = returns.corr().abs().unstack()

corr = corr.sort_values(kind="quicksort")

corr= corr[corr < 1.0][::-1]

top_n = 20

most_ticker_corr = list(set([c[0] for c in corr[:top_n].index] + [c[1] for c in corr[:top_n].index]))

small_returns = returns[most_ticker_corr]



corr = small_returns.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))



fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(corr, mask=mask, cmap="Blues", center=0, linewidths=1, annot=True, fmt=".2f", ax=ax);
year_array = returns.index.map(lambda x: int(x.year))

returns.plot.scatter("PETR3", "PETR4", c=year_array, cmap=plt.cm.viridis);
n = year_array.shape[0]

random1 = np.random.normal(0,1,n)

random2 = np.random.normal(0,1,n)

df_random = pd.DataFrame(np.stack([random1, random2]).T, columns=["R1","R2"])



df_random.plot.scatter("R1", "R2", c=year_array, cmap=plt.cm.viridis);
time_array = returns["2018-01-01":"2018-05-01"].index.map(lambda x: int(x.month))

returns["2018-01-01":"2018-05-01"].plot.scatter("BBAS3", "BBDC4", c=time_array, cmap=plt.cm.viridis);



time_array = returns["2019-01-01":"2019-05-01"].index.map(lambda x: int(x.month))

returns["2019-01-01":"2019-05-01"].plot.scatter("BBAS3", "BBDC4", c=time_array, cmap=plt.cm.viridis);



time_array = returns["2020-01-01":"2020-05-01"].index.map(lambda x: int(x.month))

returns["2020-01-01":"2020-05-01"].plot.scatter("BBAS3", "BBDC4", c=time_array, cmap=plt.cm.viridis);
corr1 = get_corr_over_time(returns,"BBAS3","BBDC4", 100)

corr2 = get_corr_over_time(returns,"ITSA4", "BBDC4", 100)

corr = pd.concat([corr1, corr2], 1)



fig, ax = plt.subplots(figsize=(12,6))

corr.plot(ax=ax);
combs = list(combinations(returns.columns, r=2))

all_corr = []

window = 60



for t1,t2 in tqdm(combs):

    corr = get_corr_over_time(returns,t1,t2, window=window)

    all_corr.append(corr)



all_corr = pd.concat(all_corr, 1)

corr_mean = all_corr.mean(1)

corr_std = all_corr.std(1)

index = all_corr.index
fig, ax = plt.subplots(figsize=(30,10))

ax.errorbar(index, corr_mean, yerr=corr_std,linewidth=2.5, elinewidth=0.15, fmt='-', label="Rolling Correlation (window = {})".format(window));

ax.legend(loc="best");

ax.set_xlabel("datetime", fontsize=14);

ax.set_ylabel("Correlation coefficient", fontsize=14);

ax.set_title("Mean rolling correlation for Ibovespa's tickers", fontsize=18);

ax.vlines(pd.to_datetime("2020-03-13"),0,1, color ="k", linestyle="--", label="Covid-19");

ax.vlines(pd.to_datetime("2008-10-06"),0,1, color ="k", linestyle=":", label="Subprime Mortgage Crisis");

ax.legend(loc="best");

plt.savefig("corr.png")