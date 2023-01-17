import pandas as pd

import numpy as np

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import os 

import datetime

from datetime import date

import string

import pandas_datareader as dr
# Creating a list of stocks used for analysis 

tech_list = ['AAPL','GOOG','AMZN','MSFT','IBM']

# Setting start and end time period

end = date.today()

start = date(end.year-1, end.month, end.day)
start
end
#For loop - retreiving data from yahoo finance data and setting as a dataframe

for stock in tech_list:   

    # Setting DataFrame as the Stock Ticker 

    globals()[stock] = dr.data.get_data_yahoo(stock, start, end)



#alternatively you can create an individual df using - AAPL = dr.data.get_data_yahoo('AAPL',start,end)    
# get the base statistics 

AAPL.sample(2)
GOOG.sample(2)
AMZN.sample(2)
MSFT.sample(2)
IBM.sample(2)
#see historical trend 

AAPL['Adj Close'].plot(legend=True,figsize=(10,4))
AAPL['Volume'].plot(legend=True,figsize=(10,4))
# concatenate all the individual dataframe into one dataframe



company_list = [AAPL,GOOG,AMZN,MSFT,IBM] # creates a consolidate list with all dataframe

company_name = ['Apple','Google','Amazon','Microsoft', 'IBM'] # column list for create a new column



# creating a consolidated dataframe and assigning company names against each dataframe using zip methoc

for company, com_name in zip(company_list, company_name): 

    company["company_name"] = com_name

    

tech_list_all = pd.concat(company_list, axis=0)

tech_list_all.sample(5)
# get a count of rows and columns

tech_list_all.shape
# Records Count by each company name

tech_list_all.groupby('company_name')['Adj Close'].count()
# moving averages for 20 days, 50 days, 100 days



ma_day = [20,50,100]



for ma in ma_day:

    for company in company_list:

        column_name = f"MA for {ma} days"

        tech_list_all[column_name] = company['Adj Close'].rolling(ma).mean()
AAPL_DF = tech_list_all.loc[tech_list_all['company_name'] == 'Apple']

AAPL_DF['Daily Return'] = AAPL_DF['Adj Close'].pct_change()
AAPL_DF['Daily Return'].plot(legend=True, figsize=(20,4), linestyle='--', marker = 'o', color = 'purple')
# Average Daily Return
sns.distplot(AAPL_DF['Daily Return'].dropna(),bins=100,color='blue')
#using pandas visualizations

AAPL['Daily Return'].hist(bins=100)
#Extracting ADJ Close colum from each dataframe and create a new dataframe for further analysis

closing_df = dr.data.get_data_yahoo(tech_list,start,end)['Adj Close']
closing_df.head()
#find daily return percentage

tech_rets = closing_df.pct_change()
tech_rets.head()
sns.jointplot('GOOG','MSFT',tech_rets,kind='scatter',color='seagreen')
# pattern to identify positive and negetive correlation pattern

from IPython.display import SVG

SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')
from numpy.random import randn

from numpy.random import seed

from scipy.stats import pearsonr



# prepare data

data1 = AAPL['Adj Close']

data2 = MSFT['Adj Close']



# calculate Pearson's correlation

corr, _ = pearsonr(data1, data2)

print('Pearsons correlation: %.3f' % corr)


# prepare data

data1 = AAPL['Adj Close']

data2 = GOOG['Adj Close']



# calculate Pearson's correlation

corr, _ = pearsonr(data1, data2)

print('Pearsons correlation: %.3f' % corr)





# prepare data

data1 = MSFT['Adj Close']

data2 = GOOG['Adj Close']



# calculate Pearson's correlation

corr, _ = pearsonr(data1, data2)

print('Pearsons correlation: %.3f' % corr)


# prepare data

data1 = AMZN['Adj Close']

data2 = GOOG['Adj Close']



# calculate Pearson's correlation

corr, _ = pearsonr(data1, data2)

print('Pearsons correlation: %.3f' % corr)


# prepare data

data1 = AMZN['Adj Close']

data2 = MSFT['Adj Close']



# calculate Pearson's correlation

corr, _ = pearsonr(data1, data2)

print('Pearsons correlation: %.3f' % corr)
sns.pairplot(tech_rets.dropna())

#using KDE, scatter and histogram

returns_fig = sns.PairGrid(tech_rets.dropna())

returns_fig.map_upper(plt.scatter)

returns_fig.map_lower(sns.kdeplot)

returns_fig.map_diag(plt.hist,bins=30)

returns_fig = sns.PairGrid(closing_df.dropna())

returns_fig.map_upper(plt.scatter)

returns_fig.map_lower(sns.kdeplot)

returns_fig.map_diag(plt.hist,bins=30)

rets = tech_rets.dropna()
area = np.pi*20



plt.scatter(rets.mean(),rets.std(), s=area)

plt.xlabel('Expected Return')

plt.ylabel('Risk')    



for label, x, y  in zip(rets.columns, rets.mean(), rets.std()):

    plt.annotate(

        label,

        xy = (x,y), xytext = (50,50),

        textcoords = 'offset points', ha = 'right', va ='bottom',

        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
sns.distplot(AAPL['Daily Return'].dropna(), bins = 100, color = 'purple')
# Empharical Quantile - 0.05 or 95% confidence level

rets['AAPL'].quantile(0.05) # 95 % confidence that the daily loss would not exceed -0.02 or 2 %, say if you invested 

# $1000 there is 95% chance that on a worst day you would not loose more than 2% 
rets['GOOG'].quantile(0.05)
rets['AMZN'].quantile(0.05)
rets['MSFT'].quantile(0.05)
rets['IBM'].quantile(0.05)
# set up time 

days = 365



# delta

dt = 1/days



#Average

mu = rets.mean()['GOOG']



# Std Deviation

sigma = rets.std()['GOOG']



def stock_monte_carlo(start_price, days, mu, sigma):

    

    price = np.zeros(days)

    price[0] = start_price

    

    shock = np.zeros(days)

    drift = np.zeros(days)

    

    for x in range(1,days):

        

        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))

        

        drift[x] = mu * dt

        

        price[x] = price[x-1] + (price[x-1]* (drift[x] + shock[x]))

    

    return price
GOOG.head()

# assign the start price of the stock to variable start_price

start_price = 1111.23
for run in range(100):

    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))



plt.xlabel('Days')

plt.ylabel('Price')

plt.title('Monte Carlo Analysis for Google')
# increase the simulation 

runs = 10000



simulations = np.zeros(runs)



for run in range(runs):

    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]
q = np.percentile(simulations,1)



plt.hist(simulations, bins = 200)



#Starting Price

plt.figtext(0.6, 0.8, s = "Start Price: $%.2f" %start_price)



# Mean ending price

plt.figtext(0.6, 0.7, "Mean Final Price: $%.2f" %simulations.mean() )



# Variance of the Price  (within 99% confidence level)

plt.figtext(0.15, 0.6, "Var(0.99): $%.2f" % (start_price - q,))



# Plot a line at the 1% quantile result

plt.axvline(x = q, linewidth = 4, color = 'r')



# Title

plt.title(u"Final Price Distribution for Google Stock after %s days" % days, weight ='bold');