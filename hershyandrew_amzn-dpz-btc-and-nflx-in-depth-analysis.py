import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
pd.read_csv('../input/portfolio_data.csv', index_col = 'Date')
mydata = pd.read_csv('../input/portfolio_data.csv', index_col = 'Date')
(mydata).plot(figsize = (25, 10));

plt.show()
mydata.iloc[0]
(mydata / mydata.iloc[0] * 100).plot(figsize = (25, 10));

plt.show()
returns = (mydata / mydata.shift(1)) - 1

returns.head
annual_returns = returns.mean() * 250

annual_returns
weights_1 = np.array([0.25, 0.25, 0.25, 0.25])

weights_2 = np.array([0.3, 0.3, 0.1, 0.3])
port_1 = np.dot(annual_returns, weights_1)

port_2 = np.dot(annual_returns, weights_2)



print (port_1)

print (port_2)
def looks(x):

    print ('The yearly discrete return of this portfolio is: ' + str(round(x,5)*100) + '%')

       

looks(port_1)

looks(port_2)
log_returns = np.log(mydata/mydata.shift(1))

log_returns
amzn_annual_return = log_returns['AMZN'].mean()*250

amzn_annual_var = log_returns['AMZN'].var() *250

amzn_annual_std = log_returns['AMZN'].std()*250**.5

print (amzn_annual_return,amzn_annual_var, amzn_annual_std)
dpz_annual_return = log_returns['DPZ'].mean()*250

dpz_annual_var = log_returns['DPZ'].var()*250

dpz_annual_std = log_returns['DPZ'].std()*250**.5

print (dpz_annual_return,dpz_annual_var, dpz_annual_std)
btc_annual_return = log_returns['BTC'].mean()*250

btc_annual_var = log_returns['BTC'].var()*250

btc_annual_std = log_returns['BTC'].std()*250**.5

print (btc_annual_return,btc_annual_var, btc_annual_std)
nflx_annual_return = log_returns['NFLX'].mean()*250

nflx_annual_var = log_returns['NFLX'].var()*250

nflx_annual_std = log_returns['NFLX'].std()*250**.5

print (nflx_annual_return,nflx_annual_var, nflx_annual_std)
n_groups = 4

returns = (amzn_annual_return, dpz_annual_return, nflx_annual_return, btc_annual_return)

std = (amzn_annual_std, dpz_annual_std, nflx_annual_std, btc_annual_std)



fig,ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.4

opacity = 0.8



rects1 = plt.bar(index, returns, bar_width,

alpha = opacity,

color ='b',

label = 'log_returns_annual')



rects2 = plt.bar(index + bar_width, std, bar_width,

alpha = opacity,

color ='g',

label = 'std_annual')



plt.xlabel('Stock')

plt.ylabel('Percent')

plt.title('Returns and Standard Deviation Annual Average (May 2013- May 2019)')

plt.xticks(index + .33,('AMZN', 'DPZ', 'NFLX', 'BTC'))

plt.legend()



plt.tight_layout()

plt.show()
cov_matrix_annual = log_returns.cov() * 250

cov_matrix_annual
corr_matrix = log_returns.corr()

corr_matrix
pfolio_vol_annual_w1 = np.dot(weights_1.T,np.dot(log_returns.cov() * 250, weights_1)) **0.5

pfolio_vol_annual_w2 = np.dot(weights_2.T,np.dot(log_returns.cov() * 250, weights_2)) **0.5



def looks_2(x):

    print ('The annual volatility of this portfolio is: ' + str(round(x,6)*100) + '%')

    

looks_2(pfolio_vol_annual_w1)

looks_2(pfolio_vol_annual_w2)



pfolio_var_annual_w1 = np.dot(weights_1.T,np.dot(log_returns.cov() * 250, weights_1))

pfolio_var_annual_w2 = np.dot(weights_2.T,np.dot(log_returns.cov() * 250, weights_2))



def looks_3(x):

    print ('The annual variance of this portfolio is: ' + str(round(x,6)*100) + '%')

    

looks_3(pfolio_var_annual_w1)

looks_3(pfolio_var_annual_w2)

dr_1 = pfolio_var_annual_w1 - (weights_1[0] ** 2 * amzn_annual_var) - (weights_1[1] ** 2 * dpz_annual_var) - (weights_1[2] ** 2 * btc_annual_var)-(weights_1[3] ** 2 * nflx_annual_var)



n_dr_1 = pfolio_var_annual_w1 - dr_1



dr_1

n_dr_1



print ('The diversifiable risk in portfolio 1 is ' + str(dr_1) + 

       'The non-diversifiable risk in portfolio 1 is ' + str(n_dr_1))
dr_2 = pfolio_var_annual_w2 - (weights_2[0] ** 2 * amzn_annual_var) - (weights_2[1] ** 2 * dpz_annual_var) - (weights_2[2] ** 2 * btc_annual_var)-(weights_2[3] ** 2 * nflx_annual_var)



n_dr_2 = pfolio_var_annual_w2 - dr_2



print ('The diversifiable risk in portfolio 1 is ' + str(dr_2) + 

       'The non-diversifiable risk in portfolio 1 is ' + str(n_dr_2))
n_groups = 2

diversifiable = (dr_1, dr_2)

non_diversifiable = (n_dr_1, n_dr_2)



fig,ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.5

opacity = 1.0



rects1 = plt.bar(index, diversifiable, bar_width,

alpha = opacity,

color ='r',

label = 'dirversifiable risk')



rects2 = plt.bar(index + bar_width, non_diversifiable, bar_width,

alpha = opacity,

color ='y',

label = 'non_diversifiable')



plt.xlabel('Portfolio')

plt.ylabel('Percent')

plt.title('Diversifiable and Non-diversifiable Risk by Portfolio')

plt.xticks(index + .33,('Portfolio_1', 'Portfolio_2'))

plt.legend()



plt.tight_layout()

plt.show()