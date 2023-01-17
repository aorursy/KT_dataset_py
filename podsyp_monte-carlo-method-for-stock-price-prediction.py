import math

import matplotlib.pyplot as plt

import numpy as np

from pandas_datareader import data
ibm = data.DataReader('IBM', 'yahoo',start='1/1/2000')
time_elapsed = (ibm.index[-1] - ibm.index[0]).days
price_ratio = (ibm['Adj Close'][-1] / ibm['Adj Close'][1])

inverse_number_of_years = 365.0 / time_elapsed

cagr = price_ratio ** inverse_number_of_years - 1

print(cagr)
vol = ibm['Adj Close'].pct_change().std()
number_of_trading_days = 252

vol = vol * math.sqrt(number_of_trading_days)
print ("cagr (mean returns) : ", str(round(cagr,4)))

print ("vol (standard deviation of return : )", str(round(vol,4)))
daily_return_percentages = np.random.normal(cagr/number_of_trading_days, vol/math.sqrt(number_of_trading_days),number_of_trading_days)+1
price_series = [ibm['Adj Close'][-1]]



for drp in daily_return_percentages:

    price_series.append(price_series[-1] * drp)
plt.plot(price_series)

plt.show()
number_of_trials = 1000

for i in range(number_of_trials):

    daily_return_percentages = np.random.normal(cagr/number_of_trading_days, vol/math.sqrt(number_of_trading_days),number_of_trading_days)+1

    price_series = [ibm['Adj Close'][-1]]



    for drp in daily_return_percentages:

        price_series.append(price_series[-1] * drp)

    

    plt.plot(price_series)

plt.show()
ending_price_points = []

larger_number_of_trials = 9001 

for i in range(larger_number_of_trials):

    daily_return_percentages = np.random.normal(cagr/number_of_trading_days, vol/math.sqrt(number_of_trading_days),number_of_trading_days)+1

    price_series = [ibm['Adj Close'][-1]]



    for drp in daily_return_percentages:

        price_series.append(price_series[-1] * drp)

    

    plt.plot(price_series)

    

    ending_price_points.append(price_series[-1])



plt.show()



plt.hist(ending_price_points,bins=50)

plt.show()
expected_ending_price_point = round(np.mean(ending_price_points),2)

print("Expected Ending Price Point : ", str(expected_ending_price_point))
population_mean = (cagr+1) * ibm['Adj Close'][-1]

print ("Sample Mean : ", str(expected_ending_price_point))

print ("Population Mean: ", str(round(population_mean,2)));

print ("Percent Difference : ", str(round((population_mean - expected_ending_price_point)/population_mean * 100,2)), "%")
top_ten = np.percentile(ending_price_points,100-10)

bottom_ten = np.percentile(ending_price_points,10);

print ("Top 10% : ", str(round(top_ten,2)))

print ("Bottom 10% : ", str(round(bottom_ten,2)))
plt.hist(ending_price_points,bins=100)

plt.axvline(top_ten,color='r',linestyle='dashed',linewidth=2)

plt.axvline(bottom_ten,color='r',linestyle='dashed',linewidth=2)

plt.axhline(ibm['Adj Close'][-1],color='g', linestyle='dashed',linewidth=2)

plt.show()