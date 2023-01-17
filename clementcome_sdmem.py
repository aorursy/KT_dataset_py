# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



sns.set_context('notebook')

plt.style.use('seaborn-darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
stock_prices = pd.read_csv('../input/daily-historical-stock-prices-1970-2018/historical_stock_prices.csv')

stock_prices.head()
stock_prices["ticker"].value_counts()
stock_AHH = stock_prices[stock_prices["ticker"] == "AHH"]

stock_AHH.head()
stock_AHH.plot(x="date", y=["open","close"], figsize=(10,6))
stock_KO = stock_prices[stock_prices["ticker"] == "KO"]

stock_KO.head()
stock_KO.plot(x="date", y=["open","close"], figsize=(10,6))
stock_JNJ = stock_prices[stock_prices["ticker"] == "JNJ"]

stock_JNJ.head()
stock_JNJ.plot(x="date", y=["open","close"], figsize=(10,6))