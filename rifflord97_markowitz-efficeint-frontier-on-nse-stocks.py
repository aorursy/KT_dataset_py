#Importing necessary libraries

import os

import pandas as pd

import numpy as np

import scipy

import matplotlib.pyplot as plt

from scipy.optimize import minimize
#The final dataframe

stocks=pd.DataFrame()

for dirname, _, filenames in os.walk('/kaggle/input'):

    #just change the 5 to 50 to utilise all 50 stocks

    for filename in filenames[:5]:

        data=pd.DataFrame()

        #reading the csv files one at a time, and querrying 'Date' and 'Close' columns using a lambda

        data = pd.read_csv(os.path.join(dirname, filename), usecols=lambda x: x in ['Date', 'Close'], parse_dates=True)

        #renaming the Close column to the Stock-name i.e., filename

        data.rename(columns = {'Close':(filename.replace('.csv',''))}, inplace = True)

        data.set_index('Date',inplace=True)

        data.index = pd.to_datetime(data.index)

        #join returns a new dataframe, not inplace of old dataframe

        stocks=stocks.join(data, how='outer')



stocks.head()
log_ret = np.log(stocks/stocks.shift(1))

log_ret.head()
np.random.seed(42)

num_ports = 10000

all_weights = np.zeros((num_ports, len(stocks.columns)))

ret_arr = np.zeros(num_ports)

vol_arr = np.zeros(num_ports)

sharpe_arr = np.zeros(num_ports)



for x in range(num_ports):

    # Weights

    weights = np.array(np.random.random(len(stocks.columns)))

    weights = weights/np.sum(weights)

    

    # Save weights

    all_weights[x,:] = weights

    

    # Expected yearly-return

    ret_arr[x] = np.sum((log_ret.mean() * weights * 252))

    

    # Expected volatility

    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))

    

    # Sharpe Ratio

    sharpe_arr[x] = ret_arr[x]/vol_arr[x]
print("Max Sharpe Ratio in the array: {}".format(sharpe_arr.max()))

print("It's location in the array: {}".format(sharpe_arr.argmax()))
print(all_weights[sharpe_arr.argmax(),:])



max_sr_ret = ret_arr[sharpe_arr.argmax()]

max_sr_vol = vol_arr[sharpe_arr.argmax()]
plt.figure(figsize=(15,8))

plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')

plt.colorbar(label='Sharpe Ratio')

plt.xlabel('Volatility')

plt.ylabel('Return')

plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # red dot

plt.show()

def get_ret_vol_sr(weights):

    weights= np.array(weights)

    ret=np.sum(log_ret.mean() * weights)*252

    vol=np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))

    sr=ret/vol

    return np.array([ret, vol, sr])



def neg_sharpe(weights):

    return get_ret_vol_sr(weights)[2]*-1



def check_sum(weights):

    return np.sum(weights)-1
#constraints

cons = ({'type':'eq', 'fun': check_sum})

n_cols=len(log_ret.columns)

#bounds

bnds = (((0,1),)*n_cols)

#initial guess

init_guess = [np.repeat((1/n_cols),n_cols)]
opt_results = minimize(neg_sharpe, init_guess, method = 'SLSQP', bounds = bnds, constraints = cons)

print(opt_results)
get_ret_vol_sr(opt_results.x)
frontier_y = np.linspace(0,0.25,200)
def minimize_volatility(weights):

    return get_ret_vol_sr(weights)[1]
frontier_x = []



for possible_return in frontier_y:

    cons = ({'type':'eq', 'fun':check_sum},

            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})

    

    result = minimize(minimize_volatility,init_guess,method='SLSQP', bounds=bnds, constraints=cons)

    frontier_x.append(result['fun'])
plt.figure(figsize=(12,8))

plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')

plt.colorbar(label='Sharpe Ratio')

plt.xlabel('Volatility')

plt.ylabel('Return')

plt.plot(frontier_x,frontier_y, 'r--', linewidth=3)

plt.savefig('cover.png')

plt.show()