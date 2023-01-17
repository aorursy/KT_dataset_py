# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Loading the bitcoin historical data

mydata = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv')



# check the columns names of the data loaded

mydata.columns



# get the description of the different data columns

mydata.describe()



# finally set up the Histogram using in this case the matplotlib library

plt.title("BTC Weighted Price Histogram")

plt.hist(mydata['Weighted_Price'])
