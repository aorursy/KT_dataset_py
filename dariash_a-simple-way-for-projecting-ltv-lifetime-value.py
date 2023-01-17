# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy import optimize

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/sq_data.csv')

print (data.shape)

data.head()
data['install_date'] = pd.to_datetime(data['install_date'],dayfirst = True)

data['pay_date'] = pd.to_datetime(data['pay_date'], dayfirst = True)

data = data.sort_values('pay_date')

def get_cum_sum(date):

    return data['sum'].where(data['pay_date']<=date).sum()



data['cum_sum'] = data['pay_date'].map(lambda x: get_cum_sum(x))

def get_users_utd(date):

    return data['user'].where(data['install_date'] <=date).count()



data['users_n_utd'] = data['pay_date'].map(lambda x: get_users_utd(x))

data['ltv'] = data['cum_sum']/data['users_n_utd'].astype(float)

data['day'] = pd.to_timedelta(data['pay_date'] - data['install_date'].min()).dt.days + 1

data['day'] = data['day'].astype(int)

data.head(10)
plt.scatter(data['day'], data['ltv'], label ='Days')

plt.scatter(data['users_n_utd'], data['ltv'], color='r', label='Players')

plt.scatter(data['cum_sum'], data['ltv'], color='g', label='Revenue')

plt.scatter(data['sum'], data['ltv'], color='b', label='Daily sales')

plt.xlabel('Days, Players, Revenue, Daily sales')

plt.ylabel('LTV')

plt.legend()
sns.pairplot(data)
ltv_data = data[['day','users_n_utd', 'cum_sum', 'ltv']]

X = ltv_data[['day', 'users_n_utd','cum_sum']]

y = ltv_data['ltv']
plt.scatter(data['day'], data['ltv'])

plt.xlabel('Days')

plt.ylabel('LTV')
X_l = X['day'].values 

Y_l = y.values



coefs_l, cov = optimize.curve_fit(lambda t,a,b: a+b*np.log(t),  X_l,  Y_l)



print (coefs_l)



def ltv_func(param):

    result = coefs_l[0] + coefs_l[1]*np.log(param)

    return result

    



ltv_90_180 = ltv_func([90., 180.])

ltv90 = round(ltv_90_180[0],2)

ltv180 = round(ltv_90_180[1],2)



print ("LTV by day 90: " + str(ltv90))

print ("LTV by day 180: " + str(ltv180))

days = np.hstack([X_l, [90, 180]])

plt.scatter(X_l,y.values)

plt.plot(days,ltv_func(days.reshape(-1, 1)))

plt.xlabel('Days')

plt.ylabel('LTV')

plt.legend(['LTV forecast', 'LTV'])
k_days = [1, 3, 7, 30]

ks= []

for k in k_days:

    k_revenue = ltv_data['cum_sum'].loc[ltv_data['day'] == k].values[0] 

    coeff = ltv180/k_revenue

    ks.append(coeff)

    print ("K" + str(k) + ": " + str(round(coeff, 2)))

k180 = ltv180/(ltv180 * ltv_data['users_n_utd'].max())
plt.title('Cumulative sales elbow-curve.')

plt.plot(k_days,ks)

plt.xlabel('Days')

plt.ylabel('K')