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

from IPython.display import display, HTML

from vol4life.vol4life.stats import test_normality_skewness

from datetime import date

import matplotlib.pyplot as plt

import seaborn as sns

from glob import glob



path_ibov =  glob('/kaggle/input/ibovespa-stocks/b3*.csv')[0]

path_selic = glob('/kaggle/input/ibovespa-stocks/selic.csv')[0]
def plot_2_returns_skew(return_1, return_2):

    mean1 = return_1.mean()

    median1 = return_1.median()

    skew1 = return_1.skew()

    kurtosis1 = return_1.kurtosis()

    mean2 = return_2.mean()

    median2 = return_2.median()

    skew2 = return_2.skew()

    kurtosis2 = return_2.kurtosis()

    



    fig, ax = plt.subplots(1,2,figsize = (20,5))

    sns.distplot(return_1, hist=True, kde=True, 

                 bins=int(180/5), color = "Green", 

                 kde_kws={'linewidth': 1.5}, ax=ax[0])

    ax[0].axvline(mean1,0, 140, ls="--", color="k", label="mean = {:.5f}".format(mean1));

    ax[0].axvline(median1,0, 140, ls=":", color="k", label="median = {:.5f}".format(median1));

    

    msg = "{} (skewness = {:.3f} | kurtosis = {:.3f})".format(return_1.name, skew1, kurtosis1)

    ax[0].set_title(msg)

    ax[0].legend(loc="best");



    sns.distplot(return_2, hist=True, kde=True, 

                 bins=int(180/5), color = "Green", 

                 kde_kws={'linewidth': 1.5}, ax=ax[1])

    ax[1].axvline(mean2,0, 140, ls="--", color="k", label="mean = {:.5f}".format(mean2));

    ax[1].axvline(median2,0, 140, ls=":", color="k", label="median = {:.5f}".format(median2));

    

    msg = "{} (skewness = {:.3f} | kurtosis = {:.3f})".format(return_2.name, skew2, kurtosis2)    

    ax[1].set_title(msg)

    ax[1].legend(loc="best");
initial_date = "2017-01-01"

today = date.today()

final_date = today.strftime("%Y-%m-%d")



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



df_sort = df.set_index(["ticker", "datetime"]).sort_index()



tss = []

for ticker in ibov:  

    ts = df_sort.xs(ticker).close

    ts.name = ticker

    tss.append(ts)



ibov_prices = pd.concat(tss,1).interpolate("linear", limit_direction="both")[initial_date:final_date]

ibov_returns = ibov_prices.pct_change().dropna() 



# subset of tickers

tickers = ["ITSA4", "BPAN4", "VVAR3"]

tss = []

for ticker in tickers:  

    ts = df_sort.xs(ticker).close

    ts.name = ticker

    tss.append(ts)



prices = pd.concat(tss,1).interpolate("linear", limit_direction="both")[initial_date:final_date]

returns = prices.pct_change().dropna()

del df_sort, df





# display

display(HTML("<h3>Price dataset</h3>"))

display(HTML("<br><b>head<b>"))

display(HTML(prices.head(2).to_html()))

display(HTML("<br><b>tail<b>"))

display(HTML(prices.tail(2).to_html()))

display(HTML("<br><b>shape = {}<b>".format(prices.shape)))
p1 = [1,0,0]

return1 = prices.dot(p1).pct_change().dropna()

return1.name = "itsa4_returns"



p2 = [0,1,0]

return2 = prices.dot(p2).pct_change().dropna()

return2.name = "bpan4_returns"



p3 = [0,0,1]

return3 = prices.dot(p3).pct_change().dropna()

return3.name = "vvar3_returns"



p4 = [0.33,0.33,0.33]

return4 = prices.dot(p4).pct_change().dropna()

return4.name = "combined_returns"
n1 = pd.Series(np.random.normal(0,1,550))

n2 = pd.Series(np.random.normal(-3,1,550))





fig, ax = plt.subplots(figsize = (12,6))

rhos = [0.25,0.75]

markers = ["darkred", "darkblue"]

for rho, m in zip(rhos,markers):



    new = pd.concat([n1.sample(frac=1-rho), n2.sample(frac=rho)])

    skew = new.skew()

    if skew > 0:

        msg =  "skewed to the right"

    elif -0.5 <= skew <= 0.5:

        msg =  "symmetric"

    else:

        msg =  "skewed to the left"

    

    sns.distplot(new, hist=False, kde=True, 

                 bins=int(180/5), color = m, 

                 label ="{}\n".format(msg) + r"$\rho$ = {:.2f} | skew = {:.2f}".format(rho, skew),

                 kde_kws={'linewidth': 4}, ax=ax)

    ax.axvline(new.mean(),0, 140, ls="--", color=m, label="mean = {:.3f}".format(rho, new.mean()));

    ax.axvline(new.median(),0, 140, ls=":", color=m, label="median = {:.3f}".format(rho, new.median()));

    

ax.legend(loc="best");



rhos = np.linspace(0,1,500)

skews = []

for rho in rhos:

    new = pd.concat([n1.sample(frac=1-rho), n2.sample(frac=rho)])

    skews.append(new.skew())



skews = pd.Series(skews, index=rhos)  

skews.name = "Skewness"

skews.index.name = r"$\rho$"



fig, ax = plt.subplots(figsize =(12,6))

skews.plot(ax=ax);

ax.axhline(0,0, 140, ls="--", color="k");

ax.legend(loc="best");
size = prices.shape[0]

time = prices.index



n1 = pd.Series(np.random.normal(0.0115,1,size))

n2 = pd.Series(np.random.normal(-0.0015,1,size))





rho = 0.25

most_positive = pd.concat([n1.sample(frac=1-rho), n2.sample(frac=rho)]).sample(frac=1,replace=False)

most_positive.index = time

most_positive.name = "most positive"



rho = 0.75

most_negative = pd.concat([n1.sample(frac=1-rho), n2.sample(frac=rho)]).sample(frac=1,replace=False)

most_negative.index = time

most_negative.name = "most negative"





df = pd.concat([most_negative, most_positive],1)

fig, ax = plt.subplots(figsize =(12,6))

df.cumsum().plot(ax=ax);

ax.axhline(0,0, 140, ls="--", color="k");



plot_2_returns_skew(most_positive, most_negative)
df = pd.concat([return1, return2, return3, return4],1)

fig, ax = plt.subplots(figsize =(12,6))

df.cumsum()[[return1.name, return2.name]].plot(ax=ax);

ax.axhline(0,0, 140, ls="--", color="k");



plot_2_returns_skew(return1, return2)



df.columns = [c.split("_")[0] for c in df.columns]

rolling = df.rolling(100).skew()[["itsa4", "bpan4"]]

fig, ax = plt.subplots(figsize=(15,5))

rolling.plot(ax=ax);

ax.axhline(0,0, 140, ls="--", color="k");

ax.legend(loc="best");

ax.set_title("Rolling Skewness (sample size = 100)");



ibov_skew = ibov_returns.skew()

subset_skew = df[["itsa4", "bpan4"]].skew()



fig, ax = plt.subplots(figsize=(6,6))

sns.boxplot(ibov_skew,orient="v",ax=ax)

sns.swarmplot(subset_skew,orient="v", color="k",ax=ax, size=15.0, label ="itsa4 and bpan4");

ax.legend(loc="best");

ax.set_title("Subset Skewness compared with ibov");
noise = pd.Series(np.random.normal(0,1,500))

noise.name = "noise"



all_ts = [noise, return1, return2, return3, return4]



for ts in all_ts:

    display(HTML(test_normality_skewness(ts).to_html()))
n1 = pd.Series(np.random.normal(0,1,550))

n2 = pd.Series(np.random.normal(0,5,550))





fig, ax = plt.subplots(figsize = (12,6))

rhos = [0.25,0.5,0.75]

markers = ["darkred", "darkgreen", "darkblue"]

for rho, m in zip(rhos,markers):



    new = pd.concat([n1.sample(frac=1-rho), n2.sample(frac=rho)])

    sns.distplot(new, hist=False, kde=True, 

                 bins=int(180/5), color = m, 

                 label =r"$\rho$ = {:.2f} | kurtosis = {:.2f}".format(rho, new.kurtosis()),

                 kde_kws={'linewidth': 4, "alpha": 0.8}, ax=ax)

    

ax.legend(loc="best");



rhos = np.linspace(0,1,500)

kurtosis = []

for rho in rhos:

    new = pd.concat([n1.sample(frac=1-rho), n2.sample(frac=rho)])

    kurtosis.append(new.kurtosis())



kurtosis = pd.Series(kurtosis, index=rhos)  

kurtosis.name = "Kurtosis"

kurtosis.index.name = r"$\rho$"



fig, ax = plt.subplots(figsize =(12,6))

kurtosis.plot(ax=ax);

ax.axhline(0,0, 140, ls="--", color="k");

ax.legend(loc="best");
### Cleaning

print('removing the repository')

subprocess.call(['rm', '-rf', REPO_NAME])