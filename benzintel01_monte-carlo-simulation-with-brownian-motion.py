# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import Lib

import math

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pandas_datareader import data

import random
stockName = 'BEM.BK'

stock = data.DataReader(stockName, 'yahoo',start='01/1/2000', end='01/12/2018')

# stock = data.DataReader(stockName, 'yahoo', start='01/1/1900')

stock.head()
# all data

time_elapsed = (stock.index[-1] - stock.index[0]).days
# Day Trade Thai

workingDays = 249



# diff row

interdayReturns = []

dataAdjPrice = pd.DataFrame(stock)

for k in range(len(dataAdjPrice['Adj Close'].values)):

    if (k > 0):

        interdayReturn = (dataAdjPrice['Adj Close'].values[k] / dataAdjPrice['Adj Close'].values[k - 1]) - 1

        interdayReturns.append(interdayReturn)

        

# dividend = μ

dividend = np.mean(interdayReturns)

# Volatility = σ

stdevS = np.std(interdayReturns, ddof=1)

# annualizedVolatility

annualizedVolatility = np.sqrt(workingDays) * stdevS
changes = []

priceSeries = [dataAdjPrice['Adj Close'].values[-1]]



for k in range(workingDays):

    if k > 1:

        day = k / (k-1)

    else:

        day = 1



    event = random.randint(-1, 1)

    a = (dividend * day)

    b = (stdevS * event * np.sqrt(day))

    changePrice = priceSeries[-1] * (a + b)

    changes.append(changePrice)



    closePrice = priceSeries[-1] + changePrice;

    priceSeries.append(closePrice)



plt.figure(figsize=(15,7))

plt.title("Montecarlo Simulation by "+ stockName, loc='center', pad=30, fontsize=30)

plt.xlabel("Working Day Trade", fontsize=20)

plt.ylabel("Price (THB)", fontsize=20)

plt.plot(priceSeries)
def simByListLoop(looping = 10):

    # finding change price

    plt.figure(figsize=(15,7))



    closingPrices = [dataAdjPrice['Adj Close'].values[-1]]

    for i in range(looping):

        changes = []

        priceSeries = [dataAdjPrice['Adj Close'].values[-1]]



        for k in range(workingDays):

            if k > 1:

                day = k / (k-1)

            else:

                day = 1



            event = random.randint(-1, 1)

            a = (dividend * day) # (μΔt)

            b = (stdevS * event * np.sqrt(day)) # (σϵ sqrt(Δt))

            changePrice = priceSeries[-1] * (a + b) # S × (μΔt + σϵsqrt(Δt))

            changes.append(changePrice)



            closePrice = priceSeries[-1] + changePrice;

            priceSeries.append(closePrice)



        closingPrices.append(priceSeries[-1])

        check = round(np.mean(closingPrices),2)

        plt.plot(priceSeries)



    plt.title("Montecarlo Simulation by " + stockName + " loop: " + str(looping) + " Time", loc='center', pad=30, fontsize=30)

    plt.xlabel("Working Day Trade", fontsize=20)

    plt.ylabel("Price (THB)", fontsize=20)

    plt.show()

    

    #plot histogram

    plt.figure(figsize=(15,7))

    plt.title("Histogram by Simulation", loc='center', pad=30, fontsize=30)

    plt.xlabel("Frequency", fontsize=20)

    plt.ylabel("Price (THB)", fontsize=20)

    plt.hist(closingPrices,bins=40)



    plt.show()

    

    mean_end_price = round(np.mean(closingPrices),2)

    print("loop: " + str(looping) + " Time")

    print("Expected price: ", str(mean_end_price))

    

    return closingPrices, mean_end_price
listLooping = [10, 100, 1000, 5000]

resultsClosePrice = []

resultsMeanPrice = []



for index in range(0, len(listLooping)):

    closingPrices, mean_end_price = simByListLoop(listLooping[index])

    resultsClosePrice.append(closingPrices)

    resultsMeanPrice.append(mean_end_price)
def histogramDraw(closingPrices, mean_end_price, time):

    #lastly, we can split the distribution into percentiles

    #to help us gauge risk vs. reward



    #Pull top 10% of possible outcomes

    top_ten = np.percentile(closingPrices,100-10)



    #Pull bottom 10% of possible outcomes

    bottom_ten = np.percentile(closingPrices,10);



    plt.figure(figsize=(15,7))

    #create histogram again

    plt.hist(closingPrices,bins=40)

    #append w/ top 10% line

    plt.axvline(top_ten,color='y', linestyle='dashed',linewidth=2)

    #append w/ bottom 10% line

    plt.axvline(bottom_ten,color='r',linestyle='dashed',linewidth=2)

    #append with current price

    plt.axhline(stock['Adj Close'][-1],color='g', linestyle='dashed',linewidth=2)

    #append with Expected price

    plt.axhline(mean_end_price ,color='g',linewidth=2)





    plt.legend(['Top 10% (' + str(top_ten) + ')',

                'Bottom 10% (' + str(bottom_ten) + ')',

                'Current price (' + str(stock['Adj Close'][-1]) +')',

                'Expected price(' + str(mean_end_price) + ')'

               ])



    plt.title("Histogram by Simulation "+ stockName + " loop: " + str(time) + " Time", loc='center', pad=30, fontsize=30)

    plt.xlabel("Frequency", fontsize=20)

    plt.ylabel("Price (THB)", fontsize=20)



    plt.show()
for index in range(0, len(listLooping)):

    histogramDraw(resultsClosePrice[index], resultsMeanPrice[index], listLooping[index])