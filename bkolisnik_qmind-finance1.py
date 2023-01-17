# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

mydata = pd.read_csv("../input/all_stocks_5yr.csv")

workset = mydata.loc[:,['date','close','volume','Name']]

flixdata = workset.loc[workset['Name'] == 'NFLX']
#flixdata = flixdata.loc[:,['date','close','volume']]
flixdata = flixdata.loc[:,['date','close']]
print(flixdata)


flixdata['date'] = pd.to_datetime(flixdata['date'], format='%Y-%m-%d', errors='coerce')

flixplot = flixdata.plot(x='date',y='close',title='Netflix Closing Price vs Date')
flixplot.set_xlabel('Date')
flixplot.set_ylabel('Closing Price ($)')

flixcloseprice = flixdata['close']
flixdata_norm = (flixcloseprice - flixcloseprice.mean())/(flixcloseprice.max() - flixcloseprice.min())

flixdata['close norm'] = flixdata_norm
print(flixdata)
#flixdata = pd.concat([flixdata['date'],flixdata_norm], axis=0, ignore_index=True)
#flixdata = flixdata['date'].join(flixdata_norm)

flixplot2 = flixdata.plot(x='date',y='close norm',title='Netflix Closing Price Norm vs Date')


# Any results you write to the current directory are saved as output.
