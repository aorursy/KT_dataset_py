# yfinance is natively available on neither Colab nor Kaggle

# So, we need to install the package

!pip install yfinance
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import datetime as dt

import yfinance as yf



# First, we look into Covid-19 Pandemic

symbols_list = ["IXJ", "^GDAXI"]

start = dt.datetime(2020,1,1)

end = dt.datetime(2020,10,12)

data = yf.download(symbols_list, start=start, end=end)
data.head()
# filter column adjusted close

df = data['Adj Close']

df.head()
# pct_change for returns

# first element is NaN, so we remove

df =df.pct_change()[1:]

df['excess_return']=df['IXJ']-df['^GDAXI']

df.head()
plt.figure(figsize=(20,10))

df['IXJ'].plot()

df['^GDAXI'].plot()

plt.ylabel("Daily returns of IXJ and ^GDAXI")

plt.title("Daily returns of IXJ and ^GDAXI")

plt.legend()

plt.show()

plt.figure(figsize=(20,10))

df['excess_return'].plot()

plt.ylabel("excess returns of IXJ on ^GDAXI")

plt.title("excess returns of IXJ on ^GDAXI")

plt.legend()

plt.show()



print(np.mean(df['excess_return']))
#Generating a permutation sample

def permutation_sample(data1, data2):

    """Generate a permutation sample from two data sets."""



    # Concatenate the data sets: data

    data = np.concatenate((data1, data2))



    # Permute the concatenated array: permuted_data

    permuted_data = np.random.permutation(data)



    # Split the permuted array into two: perm_sample_1, perm_sample_2

    perm_sample_1 = permuted_data[0:len(data1)]

    perm_sample_2 = permuted_data[len(data1):]



    return perm_sample_1, perm_sample_2
#Generating permutation replicates

def draw_perm_reps(data_1, data_2, func, size=1):

    """Generate multiple permutation replicates."""



    # Initialize array of replicates: perm_replicates

    perm_replicates = np.empty(size)



    for i in range(size):

        # Generate permutation sample

        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)



        # Compute the test statistic

        perm_replicates[i] = func(perm_sample_1, perm_sample_2)



    return perm_replicates
def diff_of_means(data_1, data_2):

    """Difference in means of two arrays."""



    # The difference of means of data_1, data_2: diff

    diff = np.mean(data_1)-np.mean(data_2)



    return diff
#Hypothesis Testing:

#H0:There is no significant difference in return compared to GDAXI during pandemic crises

#we choose difference in means as out test statistics

arr=df.to_numpy()

arr_IXJ=arr[:,0]

arr_GDAXI=arr[:,1]



empirical_diff_means = diff_of_means(arr_IXJ,arr_GDAXI)

perm_replicates = draw_perm_reps(arr_IXJ,arr_GDAXI,diff_of_means, size=10000)

# Compute p-value: p

p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

print('p-value =', p)
# 2009 Swine Flu Pandemic

# On 27 April 2009, the European Union health commissioner advised Europeans to postpone nonessential travel to the United States or Mexico. 

# On 29 April 2009, Germany had the first confirmed case

# By 19 November 2009, doses of vaccine had been administered in over 16 countries. A 2009 review by the U.S. National Institutes of Health (NIH) concluded that the 2009 H1N1 vaccine has a safety profile similar to that of seasonal vaccine.

symbols_list = ["IXJ", "^GDAXI"]

start = dt.datetime(2009,4,27)

end = dt.datetime(2009,12,31)

data = yf.download(symbols_list, start=start, end=end)



# filter column adjusted close

df = data['Adj Close']



# pct_change for returns

# first element is NaN, so we remove

df =df.pct_change()[1:]

df['excess_return']=df['IXJ']-df['^GDAXI']

plt.figure(figsize=(20,10))

df['IXJ'].plot()

df['^GDAXI'].plot()

plt.ylabel("Daily returns of IXJ and ^GDAXI")

plt.title("Daily returns of IXJ and ^GDAXI")

plt.legend()

plt.show()
plt.figure(figsize=(20,10))

df['excess_return'].plot()

plt.ylabel("excess returns of IXJ on ^GDAXI")

plt.title("excess returns of IXJ on ^GDAXI")

plt.legend()

plt.show()

print(np.mean(df['excess_return']))
#Hypothesis Testing:

#H0:There is no significant difference in return compared to GDAXI during pandemic crises

#we choose difference in means as out test statistics

arr=df.to_numpy()

arr_IXJ=arr[:,0]

arr_GDAXI=arr[:,1]



empirical_diff_means = diff_of_means(arr_IXJ,arr_GDAXI)

perm_replicates = draw_perm_reps(arr_IXJ,arr_GDAXI,diff_of_means, size=10000)

# Compute p-value: p

p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

print('p-value =', p)
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import datetime as dt

import yfinance as yf



# find the symbol (i.e., google the instrument + "yahoo finance") to any data series you are interested at 

# e.g., market/sector index ETF for your chosen country and various asset classes (e.g., Comex Gold's symbol is "GC=F")

symbols_list = ["EURUSD=X", "^GDAXI"]

start = dt.datetime(2007,1,1)

end = dt.datetime(2009,10,1)

data = yf.download(symbols_list, start=start, end=end)
data.head()
# filter column adjusted close

df = data['Adj Close']

df.head()
# pct_change for returns

# first element is NaN, so we remove

df =df.pct_change()[1:]

df.head()
plt.figure(figsize=(20,10))

df['EURUSD=X'].plot()

df['^GDAXI'].plot(alpha=0.7)

plt.ylabel("Daily returns of EURUSD=X and GDAXI")

plt.legend()

plt.show()
import statsmodels.api as sm

from statsmodels import regression



X = df["^GDAXI"]

y = df["EURUSD=X"]



# Note the difference in argument order

X = sm.add_constant(X)

model = sm.OLS(y.astype(float), X.astype(float), missing='drop').fit()

predictions = model.predict(X.astype(float)) # make the predictions by the model



# Print out the statistics

print(model.summary())

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import datetime as dt

import yfinance as yf



# find the symbol (i.e., google the instrument + "yahoo finance") to any data series you are interested at 

# e.g., market/sector index ETF for your chosen country and various asset classes (e.g., Comex Gold's symbol is "GC=F")

symbols_list = ["EURUSD=X", "^GDAXI"]

start = dt.datetime(2019,10,1)

end = dt.datetime(2020,10,1)

data = yf.download(symbols_list, start=start, end=end)
data.head()
# filter column adjusted close

df = data['Adj Close']

df.head()
# pct_change for returns

# first element is NaN, so we remove

df =df.pct_change()[1:]

df.head()
plt.figure(figsize=(20,10))

df['EURUSD=X'].plot()

df['^GDAXI'].plot()

plt.ylabel("Daily returns of EURUSD=X and GDAXI")

plt.show()

import statsmodels.api as sm

from statsmodels import regression



X = df["^GDAXI"]

y = df["EURUSD=X"]



# Note the difference in argument order

X = sm.add_constant(X)

model = sm.OLS(y.astype(float), X.astype(float), missing='drop').fit()

predictions = model.predict(X.astype(float)) # make the predictions by the model



# Print out the statistics

print(model.summary())
