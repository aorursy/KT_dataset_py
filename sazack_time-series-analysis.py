# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
amazon = pd.read_csv('../input/AMZN_2006-01-01_to_2018-01-01.csv',index_col='Date', parse_dates=['Date'])
amazon.head()
amazon['2012':'2018'].plot(subplots=True, figsize=(10,12), title="Amazon Stock from 2012 to 2018")

plt.show()
amazon['Change'] = amazon.High.div(amazon.High.shift())

amazon.Change.plot(figsize=(14,6))
amazon.High.pct_change().mul(100).plot(figsize=(14,6))
google = pd.read_csv("../input/GOOGL_2006-01-01_to_2018-01-01.csv", index_col='Date', parse_dates=['Date'])
google['2012':'2018'].plot(subplots=True, figsize=(10,12), title="Google Stock from 2012 to 2018")

plt.show()
microsoft = pd.read_csv('../input/MSFT_2006-01-01_to_2018-01-01.csv',index_col='Date', parse_dates=['Date'])
microsoft['2012':'2018'].plot(subplots=True, figsize=(10,12), title="Microsoft Stock from 2012 to 2018")

plt.show()
plt.figure(figsize=(15,8))

google.High.plot()

amazon.High.plot()

microsoft.High.plot()

plt.legend(['Google','Amazon','Microsoft'])

plt.show()
normGoogle = google.High.div(google.High.iloc[0]).mul(100)

normAmazon = amazon.High.div(amazon.High.iloc[0]).mul(100)

normMicrosoft = microsoft.High.div(microsoft.High.iloc[0]).mul(100)

plt.figure(figsize=(15,8))

normGoogle.plot()

normAmazon.plot()

normMicrosoft.plot()

plt.legend(['Google','Amazon','Microsoft'])

plt.show()