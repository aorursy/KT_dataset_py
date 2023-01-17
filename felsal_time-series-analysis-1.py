import pandas as pd

import numpy as np

import seaborn as sns

from IPython.display import display, HTML, Markdown

from datetime import date

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from glob import glob

from statsmodels.tsa.stattools import acovf

from statsmodels.tsa.stattools import acf, ccf



path_ibov =  glob('/kaggle/input/ibovespa-stocks/b3*.csv')[0]

path_usd =  glob('/kaggle/input/ibovespa-stocks/usd*.csv')[0]
df = pd.read_csv(path_ibov)

df.loc[:, "datetime"]  = pd.to_datetime(df.datetime)

df = df.set_index(["ticker", "datetime", ]).sort_index()
def plot_acf(x, lag_range, reverse=True, figsize=(12, 5),

             title_fontsize=15, xlabel_fontsize=16, ylabel_fontsize=16):

    """

    plot autocorrelation of series x

    :param x: series that we will perform the lag

    :type x: pd.Series

    :param lag_range: range of lag

    :type lag_range: int

    :param out_path: path to save figure

    :type out_path: str

    :param ccf: cross-correlation function

    :type ccf: function

    :param reverse: param to reverse lags

    :type reverse: boolean

    :param figsize: figure size

    :type figsize: tuple

    :param title_fontsize: title font size

    :type title_fontsize: int

    :param xlabel_fontsize: x axis label size

    :type xlabel_fontsize: int

    :param ylabel_fontsize: y axis label size

    :type ylabel_fontsize: int

    """



    title = "{}".format(x.name)

    lags = range(lag_range)

    ac = acf(x,fft=False,nlags=lag_range)

    sigma = 1 / np.sqrt(x.shape[0])

    fig, ax = plt.subplots(figsize=figsize)

    ax.vlines(lags, [0], ac)

    plt.plot(lags, [0] * len(lags), c="black", linewidth=1.0)

    plt.plot(lags, [2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)

    plt.plot(lags, [-2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)

    ax.set_xlabel('Lag', fontsize=xlabel_fontsize)

    ax.set_ylabel('autocorrelation', fontsize=ylabel_fontsize)

    fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.93)

    





def plot_ccf(x, y, lag_range,

             figsize=(12, 5),

             title_fontsize=15, xlabel_fontsize=16, ylabel_fontsize=16):

    """

    plot cross-correlation between series x and y

    :param x: series that we leads y on the left

    :type x: pd.Series

    :param y: series that we leads x on the right

    :type y: pd.Series

    :param lag_range: range of lag

    :type lag_range: int

    :param figsize: figure size

    :type figsize: tuple

    :param title_fontsize: title font size

    :type title_fontsize: int

    :param xlabel_fontsize: x axis label size

    :type xlabel_fontsize: int

    :param ylabel_fontsize: y axis label size

    :type ylabel_fontsize: int

    """



    title = "{} & {}".format(x.name, y.name)

    lags = range(-lag_range, lag_range + 1)

    left = ccf(y, x)[:lag_range + 1]

    right = ccf(x, y)[:lag_range]



    left = left[1:][::-1]

    cc = np.concatenate([left, right])



    sigma = 1 / np.sqrt(x.shape[0])

    fig, ax = plt.subplots(figsize=figsize)

    ax.vlines(lags, [0], cc)

    plt.plot(lags, [0] * len(lags), c="black", linewidth=1.0)

    plt.plot(lags, [2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)

    plt.plot(lags, [-2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)

    ax.set_xlabel('Lag', fontsize=xlabel_fontsize)

    ax.set_ylabel('cross-correlation', fontsize=ylabel_fontsize)    

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.93)

    

def get_lead_matrix(lead_series, fixed_series, lag_range):

    

    """

    get ccf vector of size 'max_lag' for each ts in 'lead_series' in

    relation with 'fixed_series'. All the ccf results

    are arranged in matrix format.



    :param lead_series: list of series to be lagged. 

                        All series are indexed by time.

    :type lead_series: [pd.Series]

    :param fixed_series: list of series indexed by time.

    :type fixed_series: pd.Series

    :param lag_range: range of lag

    :type lag_range: int

    :return: matrix of ccf information

    :rtype: pd.DataFrame

    """



    ccf_rows = []

    for ts in lead_series:

        merged = pd.merge_asof(ts, fixed_series,

                               left_index=True, right_index=True)



        lagged_ts = merged[ts.name]

        fixed_ts = merged[fixed_series.name]

        row = ccf(fixed_ts, lagged_ts)[:lag_range +1]

        ccf_rows.append(row)







    ccf_matrix = np.array(ccf_rows)

    ccf_matrix = pd.DataFrame(ccf_matrix,

                              columns=["lag_{}".format(i) for i in range(lag_range +1)],

                              index=[ts.name for ts in lead_series]) 

    return ccf_matrix
ticker_name = "BOVA11"

# ticker_name = "ITUB4"

# ticker_name = "VVAR3"  





ticker_ts = df.xs(ticker_name).close

ticker_ts.name = ticker_name



simple_net_return = ticker_ts.pct_change().dropna()

simple_gross_return = 1 + simple_net_return

log_return = np.log(simple_gross_return)



fig, ax = plt.subplots(1,3,figsize=(26,6))

simple_net_return.plot(ax=ax[0]);

simple_gross_return.plot(ax=ax[1]);

log_return.plot(ax=ax[2]);

ax[0].set_title("Simple Net Return\n", fontsize=18);

ax[1].set_title("Simple Gross Return\n", fontsize=18);

ax[2].set_title("Log Return\n", fontsize=18);

plt.suptitle(ticker_name, fontsize=20, y=1.1);
wnoise = pd.Series(np.random.normal(0,1,500))

wnoise.name = "white noise"

fig, ax = plt.subplots(1,2,figsize=(20,5))

wnoise.plot(ax=ax[0]);

wnoise.rolling(30).mean().plot(ax=ax[1]);

ax[0].set_title("White Noise\n", fontsize=18);

ax[1].set_title("White Noise (Moving Average)\n", fontsize=18);

steps = 500

w = np.random.normal(0,1,steps)

init1,init2 = np.random.normal(0,1,2)

xs = [init1, init2]



for i in range(2, steps):

    new = xs[i-1] - 0.9*xs[i-2] + w[i]

    xs.append(new)

    

auto_ts = pd.Series(xs)

w = pd.Series(w)

fig, ax = plt.subplots(1,2,figsize=(20,5))

auto_ts.plot(ax=ax[0]);

w.plot(ax=ax[1]);

ax[1].set_title("White Noise\n", fontsize=18);

ax[0].set_title("Autoregressive series\n", fontsize=18);
steps = 500

w = np.random.normal(0,1,steps)

drift1 = 0.2

drift2 = 0.4

drift3 = - 0.1

init = 0

x1 = [init]

x2 = [init]

x3 = [init]



for i in range(1, steps):

    new1 = drift1 + x1[i-1] + w[i]

    new2 = drift2 + x2[i-1] + w[i]

    new3 = drift3 + x3[i-1] + w[i]

    x1.append(new1)

    x2.append(new2)

    x3.append(new3)



    

rw1 = pd.Series(x1)

rw1.name = r"$\delta = {}$".format(drift1)

rw2 = pd.Series(x2)

rw2.name = r"$\delta = {}$".format(drift2)

rw3 = pd.Series(x3)

rw3.name = r"$\delta = {}$".format(drift3)

fig, ax = plt.subplots(1,1,figsize=(10,5))

rw1.plot(ax=ax);

rw2.plot(ax=ax);

rw3.plot(ax=ax);

ax.set_title("Randon walk\n", fontsize=18);

ax.legend(loc="best");
wnoise1 = pd.Series(np.random.normal(0,1,500))

wnoise2 = pd.Series(np.random.normal(0,1,500))

lp1 = wnoise1.rolling(30).sum() + 100

lp2 = wnoise2.rolling(30).sum() + 100

fig, ax = plt.subplots(1,1,figsize=(10,5))

lp1.plot(ax=ax);

lp2.plot(ax=ax);

ax.set_title("Linear Process\n", fontsize=18);

window = 30



ex1 = rw1.rolling(window).mean()

ex1.name = "random walk (non-stationary)"

ex2 = wnoise.rolling(window).mean()

ex2.name = "white noise (stationary)"



fig, ax = plt.subplots(1,1,figsize=(10,5))

ex1.plot(ax=ax);

ex2.plot(ax=ax);

ax.set_title("Rolling Mean\n", fontsize=18);

ax.legend(loc="best");
window = 60



ticker_name1 = "ITUB4"

ticker_name2 = "PETR3"  

ticker_name3 = "VALE3"  

ticker_ts1 = df.xs(ticker_name1).close

ticker_ts1.name = ticker_name1

ticker_ts2 = df.xs(ticker_name2).close

ticker_ts2.name = ticker_name2

ticker_ts3 = df.xs(ticker_name3).close

ticker_ts3.name = ticker_name3

simple_net_return1 = ticker_ts1.pct_change().dropna()

simple_net_return2 = ticker_ts2.pct_change().dropna()

simple_net_return3 = ticker_ts3.pct_change().dropna()





fig, ax = plt.subplots(1,1,figsize=(10,5))

simple_net_return1.rolling(window).mean().plot(ax=ax);

simple_net_return2.rolling(window).mean().plot(ax=ax);

simple_net_return3.rolling(window).mean().plot(ax=ax);

ax.set_title("Rolling Mean Ticker Returns \n", fontsize=18);

ax.legend(loc="best");
# sanity check for the statsmodel function

ts = wnoise

n = ts.shape[0]

test_size = 40

check1_cov = acovf(ts,fft=False)[:test_size]

check1_corr = acf(ts,fft=False)[:test_size]

for i in range(test_size):

    raw_cov = ((ts - ts.mean())*(ts.shift(i) - ts.mean())).sum() / n

    raw_corr = raw_cov / ts.var(ddof=0)

    stats_cov = check1_cov[i]

    stats_corr = check1_corr[i]

    

    test_cov = (raw_cov - stats_cov)**2

    test_corr = (raw_corr - stats_corr)**2

    assert test_cov < 1e-4

    assert test_corr < 1e-4    
plot_acf(wnoise, lag_range=100)

plot_acf(simple_net_return1, lag_range=100)

plot_acf(simple_net_return2, lag_range=100)

plot_acf(simple_net_return3, lag_range=100)
# sanity check for the statsmodel function

# ccf([FIXED SERIES], [LAGGED SERIES])





x = pd.Series(np.random.normal(0,1,100))

y = x.shift(9)

y = y.fillna(1)





x_mean =  x.mean()

y_mean =  y.mean()

std_x = x.std()

std_y = y.std()

stats_corr = ccf(y, x)

my_corr = []



for h in range(90):

    cov = (x.shift(h) - x_mean)*(y - y_mean)

    cov = cov.dropna().mean()

    corr = cov/(std_x*std_y)

    my_corr.append(corr)

    test = (corr - stats_corr[h])**2

    assert test < 1e-3, print("error: h= {} | my_corr = {:.4f} | stats_corr = {:.4f}".format(h, corr, stats_corr[h]))



my_corr = np.array(my_corr)

print(my_corr.max(), stats_corr.max())

print(my_corr.min(), stats_corr.min())

plot_ccf(x, y, 12)
a = pd.Series(np.random.normal(0,1,100))

a.name = "a"

b = a.shift(2)

b = b.fillna(1)

b.name = "a.shift(2)"

plot_ccf(a, b, 8)



a = pd.Series(np.random.normal(0,1,100))

a.name = "a"

b = a.shift(-4)

b = b.fillna(1)

b.name = "a.shift(-4)"

plot_ccf(a, b, 8)





a = pd.Series(np.random.normal(0,1,100))

a.name = "a"

b = a.shift(50) 

b = b.fillna(1)

b.name = "a.shift(50)"

plot_ccf(a, b, 60)



a = pd.Series(np.random.normal(0,1,100))

a.name = "a"

w = pd.Series(np.random.normal(0,0.1,100))

b = w

b.name = "noise"

plot_ccf(a, b, 60)
usd_brl = pd.read_csv(path_usd)

usd_brl.loc[:, "datetime"]  = pd.to_datetime(usd_brl.datetime)

usd_brl = usd_brl.set_index("datetime")["usd_brl"]

brl_usd = (1/usd_brl)

brl_usd = brl_usd.pct_change().dropna()

brl_usd.name = "brl_usd"





fig, ax = plt.subplots(1,1,figsize=(10,5))

brl_usd.plot(ax=ax);

ax.set_title("BRL/USD pct_change \n", fontsize=18);

ax.legend(loc="best");



ibov = ["ABEV3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BPAC11", "BRAP4",

        "BRDT3", "BRFS3", "BRKM5", "BRML3", "BTOW3", "CCRO3", "CIEL3", "CMIG4", "COGN3", "CRFB3",

        "CSAN3", "CSNA3", "CVCB3", "CYRE3", "ECOR3", "EGIE3", "ELET3", "ELET6", "EMBR3", "ENBR3",

        "EQTL3", "FLRY3", "GGBR4", "GNDI3", "GOAU4", "GOLL4", "HAPV3", "HGTX3", "HYPE3", "IGTA3",

        "IRBR3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "LAME4", "LREN3", "MRFG3","MGLU3",

        "MRVE3", "MULT3", "NTCO3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3",

        "RAIL3", "RENT3", "SANB11", "SBSP3", "SMLS3", "SULA11", "SUZB3", "TAEE11",

        "TIMP3", "TOTS3", "UGPA3", "USIM5", "VALE3", "VIVT4", "VVAR3", "WEGE3", "YDUQ3"]



lead_series = [] 

lead_series_dict = {}



for ticker_name in ibov:

    ticker_ts = df.xs(ticker_name).close

    ticker_ts.name = ticker_name

    simple_net_return = ticker_ts.pct_change().dropna()

    lead_series.append(simple_net_return)

    lead_series_dict[ticker_name] = simple_net_return

    

lead_m = get_lead_matrix(lead_series, brl_usd, 5)

lead_m = lead_m.sort_values("lag_1", ascending=False)

fig, ax = plt.subplots(figsize=(8,20))

ax.set_title("Tickers cross-correlation on BRL/USD pct change \n", fontsize=18)

sns.heatmap(lead_m, center=0,cmap='PuOr', linewidths=1, annot=True, fmt=".3f", ax=ax, cbar=False);

plt.xticks(rotation=45);

plt.yticks(rotation=0);
ticker_1 = "ITSA4"

ticker_2 = "BBAS3"

lag = 1



ts1 = lead_series_dict[ticker_1]

ts2 = lead_series_dict[ticker_2]

m1 = pd.merge_asof(ts1, brl_usd,

                   left_index=True, right_index=True)

plot_ccf(m1[ticker_1], m1["brl_usd"], 5)

m1.loc[:, ticker_1] = m1[ticker_1].shift(lag)

m2 = pd.merge_asof(ts2, brl_usd,

                   left_index=True, right_index=True)

plot_ccf(m2[ticker_2], m2["brl_usd"], 5)

m2.loc[:, ticker_2] = m2[ticker_2].shift(lag)



fig, ax = plt.subplots(1,2,figsize=(12,5))

ax[0].scatter(m1[ticker_1], m1["brl_usd"]);

ax[1].scatter(m2[ticker_2], m2["brl_usd"]);

ax[0].set_ylabel("BRL/USD pct_change", fontsize=14);

ax[0].set_xlabel("{} simple net return (lag = {})".format(ticker_1,lag), fontsize=14);

ax[1].set_ylabel("BRL/USD pct_change", fontsize=14);

ax[1].set_xlabel("{} simple net return (lag = {})".format(ticker_2,lag), fontsize=14);

ax[0].set_title("corr = {:.3f}\nsample size = {}".format(m1.corr().iloc[0,1], m1.shape[0]), fontsize=18)

ax[1].set_title("corr = {:.3f}\nsample size = {}".format(m2.corr().iloc[0,1], m2.shape[0]), fontsize=18)

plt.subplots_adjust(wspace=0.3)

fig.suptitle("Naive correlation analysis", fontsize=20, y=1.1);


