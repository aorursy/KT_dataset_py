# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
bitUSD = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv')
bitUSD.head()
bitUSD['Timestamp'] = bitUSD['Timestamp'].values.astype(int)
import datetime

bitUSD['Timestamp'] = [

    datetime.datetime.fromtimestamp(

    x

).strftime('%Y-%m-%d %H:%M:%S')

    for x in bitUSD['Timestamp'].values

]
nbitUSD = bitUSD.copy()

nbitUSD = nbitUSD.dropna()
%matplotlib inline

# Control the default size of figures in this Jupyter notebook

%pylab inline

pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots

 

nbitUSD["Weighted_Price"].plot(grid = True)
nbitUSD.head()
nbitUSD = nbitUSD.set_index(nbitUSD['Timestamp'])

nbitUSD = nbitUSD[["Open", "High", "Low", "Close"]]

nbitUSD.head()
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY

from matplotlib.finance import candlestick_ohlc

 

def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None):

    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays

    alldays = DayLocator()              # minor ticks on the days

    dayFormatter = DateFormatter('%d')      # e.g., 12

 

    # Create a new DataFrame which includes OHLC data for each period specified by stick input

    transdat = dat.loc[:,["Open", "High", "Low", "Close"]]

    if (type(stick) == str):

        if stick == "day":

            plotdat = transdat

            stick = 1 # Used for plotting

        elif stick in ["week", "month", "year"]:

            if stick == "week":

                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks

            elif stick == "month":

                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months

            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years

            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable

            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted

            for name, group in grouped:

                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],

                                            "High": max(group.High),

                                            "Low": min(group.Low),

                                            "Close": group.iloc[-1,3]},

                                           index = [group.index[0]]))

            if stick == "week": stick = 5

            elif stick == "month": stick = 30

            elif stick == "year": stick = 365

 

    elif (type(stick) == int and stick >= 1):

        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]

        grouped = transdat.groupby("stick")

        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted

        for name, group in grouped:

            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],

                                        "High": max(group.High),

                                        "Low": min(group.Low),

                                        "Close": group.iloc[-1,3]},

                                       index = [group.index[0]]))

 

    else:

        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')

 

 

    # Set plot parameters, including the axis object ax used for plotting

    fig, ax = plt.subplots()

    fig.subplots_adjust(bottom=0.2)

    if plotdat.index[-1] - plotdat.index[0]< pd.Timedelta('730 days'):

        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12

        ax.xaxis.set_major_locator(mondays)

        ax.xaxis.set_minor_locator(alldays)

    else:

        weekFormatter = DateFormatter('%b %d, %Y')

    ax.xaxis.set_major_formatter(weekFormatter)

 

    ax.grid(True)

 

    # Create the candelstick chart

    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),

                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),

                      colorup = "black", colordown = "red", width = stick * .4)

 

    # Plot other series (such as moving averages) as lines

    if otherseries != None:

        if type(otherseries) != list:

            otherseries = [otherseries]

        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)

 

    ax.xaxis_date()

    ax.autoscale_view()

    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

 

    plt.show()

 

pandas_candlestick_ohlc(nbitUSD)