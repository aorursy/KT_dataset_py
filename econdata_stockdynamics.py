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
data_ibm=pd.read_csv('../input/stock-dynamics/IBMStock.csv')

data_ge=pd.read_csv('../input/stock-dynamics/GEStock.csv')

data_coke=pd.read_csv('../input/stock-dynamics/CocaColaStock.csv')

data_pg=pd.read_csv('../input/stock-dynamics/ProcterGambleStock.csv')

data_boe=pd.read_csv('../input/stock-dynamics/BoeingStock.csv')
data_ibm['Date'] = pd.to_datetime(data_ibm['Date'])
data_ibm['Date'] = data_ibm['Date'].dt.strftime('%m/%d/%Y')
print(data_ibm)
data_pg['Date'] = pd.to_datetime(data_coke['Date'])

data_pg['Date'] = data_pg['Date'].dt.strftime('%m/%d/%Y')

print(data_pg)
data_ge['Date'] = pd.to_datetime(data_ge['Date'])

data_ge['Date'] = data_ge['Date'].dt.strftime('%m/%d/%Y')

print(data_ge)
data_coke['Date'] = pd.to_datetime(data_ibm['Date'])

data_coke['Date'] = data_coke['Date'].dt.strftime('%m/%d/%Y')

print(data_coke)
data_boe['Date'] = pd.to_datetime(data_ibm['Date'])

data_boe['Date'] = data_boe['Date'].dt.strftime('%m/%d/%Y')

print(data_boe)
def f(x):

    x=pd.to_datetime(x)

    return x.dt.strftime('%m/%d/%Y')

#write a function that does a job in one go
f(data_boe['Date'])
data_boe
#Our five datasets all have the same number of observations. How many observations are there in each data set?

#480
data_boe.tail()
#What is the earliest year in our datasets?

#the minimum value of the Date variable is January 1, 1970 for any dataset.

#What is the latest year in our datasets?

#the maximum value of the Date variable is December 1, 2009 for any dataset.
data_ibm.mean()
#What is the mean stock price of IBM over this time period?

# 144.37503
data_ge.min()
#What is the min stock price of GE over this time period?

# 9.29364
data_coke.max()
#What is the max stock price of coke over this time period?

# 146.584
data_boe.median()
#what is the median stock price of boeing stock over this time period

#44.88

data_pg.std()
#What is the sd stock price of GE over this time period?

# 18.19414
#data visualization
import matplotlib.pyplot as plt
plt.plot(data_coke.Date,data_coke.StockPrice)
from matplotlib import pyplot

series = pd.read_csv('../input/stock-dynamics/CocaColaStock.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series1 = pd.read_csv('../input/stock-dynamics/ProcterGambleStock.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series1.plot()

series.plot()

pyplot.show()
def viz(path):

    series=pd.read_csv(path,header=0,index_col=0,parse_dates=True,squeeze=True)

    series.plot()

    pyplot.show()
path='../input/stock-dynamics/CocaColaStock.csv'

path2='../input/stock-dynamics/ProcterGambleStock.csv'

viz(path)

viz(path2)

pyplot.show()
#Around what year did Coca-Cola has its highest stock price in this time period?

#1973
#Around what year did Coca-Cola has its lowest stock price in this time period?

#1980
from matplotlib import pyplot

series = pd.read_csv('../input/stock-dynamics/CocaColaStock.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series1 = pd.read_csv('../input/stock-dynamics/ProcterGambleStock.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series1.plot()

series.plot()

pyplot.show()
#In March of 2000, the technology bubble burst, and a stock market crash occurred. According to this plot, which company's stock dropped more?

#P&G



#Around 1983, the stock for one of these companies (Coca-Cola or Procter and Gamble) was going up, while the other was going down. Which one was going up?

#coca cola
#In the time period shown in the plot, which stock generally has lower values?

#Looking at the plot, the red line (for Coca-Cola) is generally lower than the blue line.



path1='../input/stock-dynamics/GEStock.csv'

path2='../input/stock-dynamics/CocaColaStock.csv'

path3='../input/stock-dynamics/IBMStock.csv'

path4='../input/stock-dynamics/ProcterGambleStock.csv'

path5='../input/stock-dynamics/BoeingStock.csv'

viz(path1)

viz(path2)

viz(path3)

viz(path4)

viz(path5)
#Which stock fell the most right after the technology bubble burst in March 2000?

#GE
viz(path1)
viz(path2)
viz(path3)
viz(path4)
viz(path5)
def f(n):

    if n==0:

        return 0

    elif n==1:

        return n

    else:

        return f(n-1)+f(n-2)