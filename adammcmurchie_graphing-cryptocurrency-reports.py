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
# ======================== Python dependencies

import os

import random

import pandas

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter

from pylab import figure, axes, pie, title, show

import datetime

from datetime import timedelta





"""



THIS JOB READS FROM REPORTS FOLDER, EXTRACTS DATA, PRODUCES GRAPHS SAVES AS PNG TO A LOCATION OF YOUR CHOICE/



"""
# ========================  Create Array of market names from available files 



directory = "/kaggle/input/cryptocurrency-rates-ingbp/"

#initialize iterators

availableMarkets = []

availableMarketsFile = []

metricArray = []

metricRow = []

coinmetricRow = []







for filename in os.listdir(directory):

    if filename.endswith(".txt"): 

        #print(os.path.join(directory, filename))

        marketName = filename[:-14]

        availableMarketsFile.append(filename)

        availableMarkets.append(marketName)

        continue

    else:

        continue

print('Printing available Markets...')

print(availableMarkets)
# ======================== Create populated Markets matrix



print('Importing Market Data')

for j in range(0, len(availableMarkets)): # Iterate through files

    with open(directory + str(availableMarketsFile[j]), 'rU') as f:

        metricRow = []

        print(availableMarketsFile[j])

        for line in f: 

            line = line[:-1]

            words = line.split(",")

            #print(words)

            marketArray = np.asarray(words)

            metricRow.append(marketArray)

        coinmetricRow.append(np.asarray(metricRow))
for y in range (0, len(coinmetricRow)):

    rates = []

    dates = []

    for x in range(0, len(coinmetricRow[y])):

        rates.append(float(coinmetricRow[y][x][2][1:]))

        toDatetime = datetime.datetime.strptime(coinmetricRow[y][x][0][:-3], "%Y-%m-%d %H:%M")

        dates.append(toDatetime)





    plt.clf() # clear cache

    color = ["blue","red","green","black","orange"]

    print(rates)

    fig, ax = plt.subplots()

    ax.plot(dates,rates, color=random.choice(color))

    fig.autofmt_xdate()

    myFmt = DateFormatter("%Y-%m-%d")

    ax.xaxis.set_major_formatter(myFmt)

    plt.title(str(coinmetricRow[y][0][1]) + " to £GBP", fontsize=20)

    plt.ylabel("pounds GBP")  

    plt.show()

    print('Saving Output')

    plt.savefig(str(coinmetricRow[y][0][1]) + "-report.png", bbox_inches="tight")