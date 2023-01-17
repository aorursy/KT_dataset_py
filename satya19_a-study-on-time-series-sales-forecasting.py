# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data visualization

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib import style

plt.rcParams['figure.figsize']=(20,10) # set the figure size

plt.style.use('fivethirtyeight') # using the fivethirtyeight matplotlib theme



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



sales = pd.read_csv("../input/sales.csv")



sales.head()



# Any results you write to the current directory are saved as output.
sales.isnull().sum()
plt.boxplot(sales.Sales_Dollars)
plt.boxplot(sales.Quantity)
sales.Date = pd.to_datetime(sales.Date) #set the date column to datetime

sales.set_index('Date', inplace=True) #set the index to the date column

sales.head()
#This is for Display each month total sales in Dollars and total quantity

# I am taking a year from oct-oct one year

#-----------------------------------------------------------------------

# now the hack for the multi-colored bar chart: 

# create fiscal year dataframes covering the timeframes you are looking for. In this case,

# the fiscal year covered October - September.



# --------------------------------------------------------------------------------

# Note: This should be set up as a function, but for this small amount of data,

# I just manually built each fiscal year. This is not very pythonic and would

# suck to do if you have many years of data, but it isn't bad for a few years of data. 

# --------------------------------------------------------------------------------

 

fy10_all = sales[(sales.index >= '2009-10-01') & (sales.index < '2010-10-01')]

fy11_all = sales[(sales.index >= '2010-10-01') & (sales.index < '2011-10-01')]

fy12_all = sales[(sales.index >= '2011-10-01') & (sales.index < '2012-10-01')]

fy13_all = sales[(sales.index >= '2012-10-01') & (sales.index < '2013-10-01')]

fy14_all = sales[(sales.index >= '2013-10-01') & (sales.index < '2014-10-01')]

fy15_all = sales[(sales.index >= '2014-10-01') & (sales.index < '2015-10-01')]

 

# Let's build our plot

 

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()  # set up the 2nd axis

ax1.plot(sales.Sales_Dollars) #plot the Revenue on axis #1

 

# the next few lines plot the fiscal year data as bar plots and changes the color for each.

ax2.bar(fy10_all.index, fy10_all.Quantity,width=20, alpha=0.2, color='orange')

ax2.bar(fy11_all.index, fy11_all.Quantity,width=20, alpha=0.2, color='gray')

ax2.bar(fy12_all.index, fy12_all.Quantity,width=20, alpha=0.2, color='blue')

ax2.bar(fy13_all.index, fy13_all.Quantity,width=20, alpha=0.2, color='red')

ax2.bar(fy14_all.index, fy14_all.Quantity,width=20, alpha=0.2, color='orange')

ax2.bar(fy15_all.index, fy15_all.Quantity,width=20, alpha=0.2, color='gray')

handles, labels =ax1.get_legend_handles_labels()

ax1.legend(handles, labels)

 

ax2.grid(b=False) # turn off grid #2

 

ax1.set_title('Monthly Sales Revenue vs Number of Items Sold Per Month')

ax1.set_ylabel('Monthly Sales Revenue')

ax2.set_ylabel('Number of Items Sold')

 

# Set the x-axis labels to be more meaningful than just some random dates.

labels = ['FY 2010', 'FY 2011','FY 2012', 'FY 2013','FY 2014', 'FY 2015']

ax1.axes.set_xticklabels(labels)
# Second analytics way

sales1 = pd.read_csv("../input/sales.csv") 
#We have a good 6-year sales data.

sales1['Date'].min(), sales1['Date'].max()
latest = sales1.Date.max()

oldest = sales1.Date.min()

A = sales1[sales1.Date >= latest]

B = sales1[sales1.Date <= oldest]

C = (float(A.Sales_Dollars)-float(B.Sales_Dollars))/float(B.Sales_Dollars)

growth =  float(C) * 100

print('Overall Growth:', growth, "%")
sales1['Date'] = pd.to_datetime(sales1['Date']) #Again convert date into datetime

sales1 = sales1.groupby('Date').sum()  # Groupby date and sum both column
y = sales1['Sales_Dollars'].resample('MS').mean()  

y.plot(figsize=(15, 6))

plt.show()
from pylab import rcParams

import statsmodels.api as sm

rcParams['figure.figsize'] = 18, 8



decomposition = sm.tsa.seasonal_decompose(y, model='additive')

fig = decomposition.plot()

plt.show()

#The plot above clearly shows that the sales is unstable, along with its obvious seasonality.
import warnings

import itertools

warnings.filterwarnings("ignore")



p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['text.color'] = 'k'

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False, enforce_invertibility=False)

                                                                                



            results = mod.fit()



            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
mod = sm.tsa.statespace.SARIMAX(y,

                                order=(1, 1, 1),

                                seasonal_order=(1, 1, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)



results = mod.fit()



print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('2015-09-01'), dynamic=False)
y_forecasted = pred.predicted_mean

y_truth = y['2015-09-01':]



mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=50)

pred_ci = pred_uc.conf_int()



ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Sales')



plt.legend()

plt.show()