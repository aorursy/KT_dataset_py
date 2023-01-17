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

import seaborn as sns

from scipy.stats import norm

from IPython.display import display, HTML, Markdown

from datetime import date

from tqdm import tqdm

from vol4life.vol4life.style import show_red_green



import matplotlib.pyplot as plt

from glob import glob



path_ibov =  glob('/kaggle/input/ibovespa-stocks/b3*.csv')[0]

path_usd =  glob('/kaggle/input/ibovespa-stocks/usd*.csv')[0]

path_selic =  glob('/kaggle/input/ibovespa-stocks/selic.csv')[0]

path_sp = glob("/kaggle/input/sp-500-full-dataset/*.csv")
df = pd.read_csv(path_usd)

df.loc[:, "datetime"]  = pd.to_datetime(df.datetime)

df = df.set_index("datetime")

usd2brl = df.copy() 

usd = (1/df)

usd.columns = ["brl2usd"]

usd = usd.sort_index()

usd_returns  = df.pct_change().dropna()

usd_returns.name = "usd_returns"

fig, ax = plt.subplots(figsize=(10,5))

usd.plot(ax=ax);
df = pd.read_csv(path_ibov)

df.loc[:, "datetime"] =  pd.to_datetime(df.datetime)



ibov = ["ABEV3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BPAC11", "BRAP4",

        "BRDT3", "BRFS3", "BRKM5", "BRML3", "BTOW3", "CCRO3", "CIEL3", "CMIG4", "COGN3", "CRFB3",

        "CSAN3", "CSNA3", "CVCB3", "CYRE3", "ECOR3", "EGIE3", "ELET3", "ELET6", "EMBR3", "ENBR3",

        "EQTL3", "FLRY3", "GGBR4", "GNDI3", "GOAU4", "GOLL4", "HAPV3", "HGTX3", "HYPE3", "IGTA3",

        "IRBR3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "LAME4", "LREN3", "MRFG3",

        "MRVE3", "MULT3", "NTCO3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3",

        "RAIL3", "RENT3", "SANB11", "SBSP3", "SMLS3", "SULA11", "SUZB3", "TAEE11",

        "TIMP3", "TOTS3", "UGPA3", "USIM5", "VALE3", "VIVT4", "VVAR3", "WEGE3", "YDUQ3"]





# sem "MGLU3", muito estranho



initial_date = "2015-01-01"

today = date.today()

final_date = today.strftime("%Y-%m-%d")





df_sort = df.set_index(["ticker", "datetime"]).sort_index()

tss = []

for ticker in ibov:  

    ts = df_sort.xs(ticker).close

    ts.name = ticker

    tss.append(ts)



del df_sort

prices = pd.concat(tss,1).interpolate("linear", limit_direction="both")[initial_date:final_date]



usd = usd[initial_date:final_date]

ibov_usd = pd.concat([prices, usd],1).dropna().interpolate("linear", limit_direction="both")

rate = ibov_usd.brl2usd

ibov_usd = ibov_usd.multiply(rate, 0)

ibov_usd = ibov_usd.drop("brl2usd",1)



ibov_prices_usd = ibov_usd.mean(1)

ibov_prices_brl = prices.mean(1)





fig, ax = plt.subplots(figsize=(15,5))

ibov_prices_usd.plot(ax=ax, color="k", label="ibov_usd");

ax.set_title('Ibov in USD',fontsize=14);
sp_closes = []

for path in tqdm(path_sp):

    df =  pd.read_csv(path)

    df.loc[:, "Date"]  = pd.to_datetime(df.Date)

    sp_closes.append(df.set_index("Date").Close)

    

sp = pd.concat(sp_closes,1).interpolate("linear", limit_direction="both")[initial_date:final_date]

sp = sp.mean(1)

sp.name = "S&P 500"



sp_brl = pd.concat([sp, usd2brl[initial_date:final_date]],1).dropna().interpolate("linear", limit_direction="both")

sp_brl = sp_brl['S&P 500'] * sp_brl['usd_brl']

sp_brl.name = "S&P 500"





fig, ax = plt.subplots(figsize=(15,5))

sp.plot(ax=ax, color="k", label="ibov_usd");

ax.set_title("S&P 500",fontsize=14);
window = 30

fig, ax = plt.subplots(2,2, figsize=(24,10))

ibov_prices_usd.rolling(window).mean().plot(ax=ax[0,0], color="k",  style=":", label="Ibov");

sp.rolling(window).mean().plot(ax=ax[0,0], color="k", style="-", label="S&P 500");

ax[0,0].set_title("{} Day Rolling Mean".format(window),fontsize=14);



ibov_prices_usd.rolling(window).std().plot(ax=ax[0,1], color="k",  style=":", label="Ibov");

sp.rolling(window).std().plot(ax=ax[0,1], color="k", style="-", label="S&P 500");

ax[0,1].set_title("{} Day Rolling Standard Deviation".format(window),fontsize=14);





ibov_prices_usd.rolling(window).skew().plot(ax=ax[1,0], color="k",  style=":", label="Ibov");

sp.rolling(window).skew().plot(ax=ax[1,0], color="k", style="-", label="S&P 500");

ax[1,0].set_title("{} Day Rolling Skewness".format(window),fontsize=14);



ibov_prices_usd.rolling(window).kurt().plot(ax=ax[1,1], color="k",  style=":", label="Ibov");

sp.rolling(window).kurt().plot(ax=ax[1,1], color="k", style="-", label="S&P 500");

ax[1,1].set_title("{} Day Rolling Kurtosis".format(window),fontsize=14);

ax[1,1].legend(loc="upper left");

plt.subplots_adjust(hspace=0.4, wspace=0.1)

plt.figtext(0.5, 0.95, 'Price Statistics for Tickers in the S&P 500 and Ibovespa', ha='center', va='center',fontsize=18);
from matplotlib import figure

from matplotlib.backends.backend_agg import FigureCanvasAgg

from matplotlib.ticker import FuncFormatter





def f_percentage(x, pos):

    """

    Adds percentage sign to plot ticks.

    """



    return '%.0f%%' % x



def cum_returns(returns, starting_value=0):

    """

    Compute cumulative returns from simple returns.

    Parameters

    """

    creturns = returns.copy()

    creturns = creturns.fillna(0)

    creturns = (creturns + 1).cumprod()

    if starting_value == 0:

        creturns = creturns - 1

    else:

        creturns = creturns * starting_value

    return creturns





def plot_drawdown_underwater(returns, ax=None, title='Underwater plot', **kwargs):

    """

    Plots how far underwaterr returns are over time, or plots current

    drawdown vs. date.

    Parameters

    ----------

    returns : pd.Series

        Daily returns of the strategy, noncumulative.

         - See full explanation in tears.create_full_tear_sheet.

    ax : matplotlib.Axes, optional

        Axes upon which to plot.

    **kwargs, optional

        Passed to plotting function.

    Returns

    -------

    ax : matplotlib.Axes

        The axes that were plotted on.

    """



    if ax is None:

        ax = plt.gca()



    y_axis_formatter = FuncFormatter(f_percentage)

    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))



    df_cum_rets = cum_returns(returns, starting_value=1.0)

    running_max = np.maximum.accumulate(df_cum_rets)

    underwater = -100 * ((running_max - df_cum_rets) / running_max)

    (underwater).plot(ax=ax, kind='area', color='coral', alpha=0.7, **kwargs)

    ax.set_ylabel('Drawdown', fontsize=14)

    ax.set_title(title, fontsize=18)

    ax.set_xlabel('')

    return ax
r_usd = ibov_prices_usd.pct_change().dropna()

r_sp = sp.pct_change().dropna()



fig, ax = plt.subplots(2, figsize=(20,10))

plot_drawdown_underwater(r_usd["2018-12-30":], ax[1], title="Underwater plot (Ibov in USD)");

plot_drawdown_underwater(r_sp["2018-12-30":], ax=ax[0], title="Underwater plot (S&P 500)");

fig.tight_layout(pad=3.0);
usd = usd_returns[initial_date:final_date]

r_brl = r_brl = ibov_prices_brl.pct_change().dropna().to_frame()



r_sp_brl = sp_brl.pct_change().dropna()

usd.columns = ["US Dollar"]

r_brl.columns = ["Ibov"]

r_sp_brl.columns = ["S&P 500"]



selic = pd.read_csv(path_selic)

selic.datetime = pd.to_datetime(selic.datetime)

selic = selic.set_index("datetime")

selic = selic[initial_date:final_date]

selic.columns = ["Selic"]



cum_selic = cum_returns(selic, starting_value=1.0)

cum_r_brl = cum_returns(r_brl, starting_value=1.0)

cum_usd = cum_returns(usd, starting_value=1.0)

cum_r_sp_brl = cum_returns(r_sp_brl, starting_value=1.0)







fig, ax = plt.subplots(figsize=(15,8))

ax.set_title("Cummulative returns by instrument (in BRL)\n", fontsize=18)

ax.set_ylabel('cummulative return (percentage)', fontsize=14)

y_axis_formatter = FuncFormatter(lambda x, pos: "{:.1%}".format(x))

ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

cum_usd.plot(ax=ax);

cum_r_brl.plot(ax=ax);

cum_selic.plot(ax=ax);

cum_r_sp_brl.plot(ax=ax);

ax.legend(loc="best");
df = pd.read_csv(path_ibov)

df.loc[:, "datetime"] =  pd.to_datetime(df.datetime)



etfs =     ["BOVA11",

            "BOVV11",

            "SMAL11",

            "BOVB11",

            "IVVB11",

            "SMAC11",

            "DIVO11",

            "PIBB11",

            "BRAX11",

            "SPXI11",

            "FIND11",

            "MATB11",

            "ISUS11",

            "GOVE11",

            "ECOO11",

            "BBSD11",

            "XBOV11"]





initial_date = "2019-01-01"

today = date.today()

final_date = today.strftime("%Y-%m-%d")





df_sort = df.set_index(["ticker", "datetime"]).sort_index()

tss = []

for ticker in etfs:  

    ts = df_sort.xs(ticker).close

    ts.name = ticker

    tss.append(ts)



del df_sort

prices = pd.concat(tss,1).interpolate("linear", limit_direction="both")[initial_date:final_date]

total  = prices.pct_change().dropna()



total = total.apply(lambda row: cum_returns(row, starting_value=1.0))





fig, ax = plt.subplots(figsize=(22,13))

ax.set_title("Cummulative returns by ETF (in BRL)\n", fontsize=18)

ax.set_ylabel('cummulative return (percentage)', fontsize=14)

y_axis_formatter = FuncFormatter(lambda x, pos: "{:.1%}".format(x))

ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

total.plot(ax=ax, cmap='tab20');

ax.legend(loc="lower left");



corr = prices.tail(60).corr()

mask = np.triu(np.ones_like(corr, dtype=bool))



fig, ax = plt.subplots(figsize=(14,10))

ax.set_title("60-day correlation Brazilian ETFs\n", fontsize=18)

sns.heatmap(corr, mask=mask, cmap="Blues", center=0, linewidths=1, annot=True, fmt=".2f", ax=ax, cbar=False);

plt.xticks(rotation=45);
month_dict  =  {1:"January",

                2:"February",

                3:"March",

                4:"April",

                5:"May",

                6:"June",

                7:"July",

                8:"August",

                9:"September",

                10:"October",

                11:"November",

                12:"December"}





month_etf = prices["2019-12-01":].resample("M").last()

etf_columns = list(month_etf.columns)

etf_columns.sort()

month_etf = month_etf[etf_columns]

rate_etf = pd.merge_asof(month_etf, rate, left_index=True, right_index=True)["brl2usd"]

for c in etf_columns:

    month_etf.loc[:,c] = month_etf[c] * rate_etf



month_etf = month_etf.pct_change()

month_etf = month_etf.dropna()

month_etf.index.name = ""

month_etf.index = [month_dict[m.month] for m in month_etf.index]

display(Markdown("## Monthly return in 2020 by ETF (in USD)"))

show_red_green(month_etf, 1)
### Cleaning

print('removing the repository')

subprocess.call(['rm', '-rf', REPO_NAME])