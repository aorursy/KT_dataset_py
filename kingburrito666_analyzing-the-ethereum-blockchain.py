import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt # converting timestamp to date
import seaborn as sns # Visualizer for data
import matplotlib.pyplot as plt # matplot
import matplotlib.dates as mdates # for plotting dates
%matplotlib inline
from subprocess import check_output
# What are we working with?
print('File: \n', check_output(["ls", "../input"]).decode("utf8"))
# Input files i'm using
address_gr = pd.read_csv('../input/EthereumUniqueAddressGrowthRate.csv')
blocksize_hist = pd.read_csv('../input/EthereumBlockSizeHistory.csv')
etherprice_usd = pd.read_csv('../input/EtherPriceHistory(USD).csv')
hashrate_gr = pd.read_csv('../input/EthereumNetworkHashRateGrowthRate.csv')
marketcap = pd.read_csv('../input/EtherMarketCapChart.csv')
tx_hist = pd.read_csv('../input/EthereumTransactionHistory.csv')
# Going to iterate and plot everything, except those with abnormalities
things_to_plot = [(blocksize_hist,"Blocksize History"),
                  (etherprice_usd, "Etherprice - USD"),
                  (hashrate_gr,"Hashrate Growth Rate"),
                # (address_gr, "Address Growth Rate"),
                # (marketcap, "Market Capital"),
                  (tx_hist, "Transaction History")]
# the timestamp in the method is a dataframe column
# it returns a list of the format which can then be ploted if needed
def timeConvert(timestamps):
    timeValue = list(range(len(timestamps)))
    for i in range(len(timestamps)):
        timeValue[i] = (dt.datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d'))
    return timeValue;
# Lets see:
print(marketcap.columns)
def plotit(data, title):
    # makes numpy array
    r = data.values#.view(np.recarray)
    #grab dates - convert to format
    date_df = r[:,0]
    date_df = pd.to_datetime(date_df)
    #grab values
    value_df = r[:,2]
    # make new plots
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_title(title)
    ax.plot(date_df, value_df)
    ax.grid(False)
    # matplotlib date format object
    hfmt = mdates.DateFormatter('%Y - - %m')
    # format the ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(hfmt)
    # format the coords message box
    def yvals(x):
        return '$%1.2f' % x
    ax.format_xdata = hfmt
    ax.format_ydata = yvals
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.show()
plotit(etherprice_usd, "Etherprice - USD")
for plot,title in things_to_plot:
    plotit(plot, title)
mkp = marketcap.values#.view(np.recarray)

date_df = mkp[:,0]
date_df = pd.to_datetime(date_df)
value_df = mkp[:,3]
prices_df = mkp[:,4]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(15, 7))
ax.set_title("Market Capital")
ax.set_ylabel("(USD) Millions")
ax.plot(date_df, value_df)
ax.grid(False)
# Format dates
hfmt = mdates.DateFormatter('%Y - - %m')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(hfmt)
def yvals(x):
    return '$%1.2f' % x
ax.format_xdata = hfmt
ax.format_ydata = yvals
fig.autofmt_xdate()
plt.show()
txs = tx_hist.copy()
txs['Date(UTC)'] = pd.to_datetime(txs['Date(UTC)']).dt.year
#txs['Date(UTC)'] = txs['Date(UTC)'].dt.year
txs = txs.groupby('Date(UTC)')['Value'].apply(lambda x: (x.unique().sum()))
txs
fig, ax = plt.subplots(figsize=(10, 10))
shap = txs
labels = '2015','2016','2017','2018'
explode = (0, 0, 0, 0.1)
ax.pie(shap, explode=explode, labels=labels, shadow=True)
plt.title('Transactions per year')
plt.show()