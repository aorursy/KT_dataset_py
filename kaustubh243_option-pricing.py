!pip install QuantLib
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import datetime as dt
from pandas_datareader.yahoo.options import Options as YahooOptions
import time
from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
import tensorflow as tf
from statistics import mean
from math import sqrt,pi,log
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot 
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from scipy.interpolate import make_interp_spline, BSpline
from scipy.integrate import simps, cumtrapz, romb
import QuantLib as ql
options_table = pd.read_csv('/kaggle/input/aapl-options-data/options-data - option_tick_processed_20121005.csv')
options_table1 = pd.read_csv('/kaggle/input/options/option_tick_processed_20121031.txt', sep="	")
options_table2 = pd.read_csv('/kaggle/input/options1/option_tick_processed_20121010.txt/option_tick_processed_20121010.txt', sep="	")
options_table3 = pd.read_csv('/kaggle/input/options1/option_tick_processed_20121003.txt/option_tick_processed_20121003.txt', sep="	")
options_table4 = pd.read_csv('/kaggle/input/options-2/option_tick_processed_20121011.txt/option_tick_processed_20121011.txt', sep="	")
options_table5 = pd.read_csv('/kaggle/input/options-2/option_tick_processed_20121004.txt/option_tick_processed_20121004.txt', sep="	")
options_table6 = pd.read_csv('/kaggle/input/options-2/option_tick_processed_20121001.txt/option_tick_processed_20121001.txt', sep="	")
options_table7 = pd.read_csv('/kaggle/input/options-2/option_tick_processed_20121024.txt/option_tick_processed_20121024.txt', sep="	")
options_table01 = options_table.append(options_table1, ignore_index = True)
options_table02 = options_table01.append(options_table2, ignore_index = True)
options_table03 = options_table02.append(options_table3, ignore_index = True)
options_table04 = options_table03.append(options_table4, ignore_index = True)
options_table05 = options_table04.append(options_table5, ignore_index = True)
options_table06 = options_table05.append(options_table6, ignore_index = True)
options_table = options_table06.append(options_table7, ignore_index = True)
options_table.drop_duplicates(inplace = True)
options_table.info()
options_table.head()
print(len(options_table))
P_options_index = options_table[options_table['OptionType'] == 'P'].index
options_table= options_table.drop(P_options_index)

options_table.head(1)

options_table.info()

def call_payoff(S , K):
    return np.maximum(S-K, 0)
def N(x):
    return norm.cdf(x)

def bs_C_value(S,K,r,t,v):
    d1 = (1.0/(v * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * v**2.0) * t)
    d2 = d1 - (v * np.sqrt(t))
    return N(d1) * S - N(d2) * K * np.exp(-r * t)

def bs_P_value(S,K,r,t,v):
    d1 = (1.0/(v * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * v**2.0) * t)
    d2 = d1 - (v * np.sqrt(t))
    return  N(-d2) * K * np.exp(-r * t) - N(-d1) * S 

def call_iv_char(S,K,r,t,v,call_price):
    return call_price - bs_C_value(S,K,r,t,v)

def put_iv_char(S,K,r,t,v,put_price):
    return put_price - bs_P_value(S,K,r,t,v)

def C_iv(S,K,r,t,call_price, a = -2.0, b = 2.0, tol = 1e-6):
    S_1 = S
    K_1 = K
    r_1 = r
    t_1 = t
    call_price_1 = call_price

    def fun(v):
        return call_iv_char(S_1,K_1,r_1,t_1,v,call_price_1)
    try:
        res = brentq(fun, a = a, b = b, xtol = tol)
        return np.nan if res <=1.0e-6 else res
    except ValueError:
        return np.nan
def P_iv(S,K,r,t,put_price, a = -2.0, b = 2.0, tol = 1e-6):
    S_1 = S
    K_1 = K
    r_1 = r
    t_1 = t
    put_price_1 = put_price

    def fun(v):
        return put_iv_char(S_1,K_1,r_1,t_1,v,put_price_1)
    try:
        res = brentq(fun, a = a, b = b, xtol = tol)
        return np.nan if res <=1.0e-6 else res
    except ValueError:
        return np.nan
def getput(x):
    S = x['StockLast']
    K = x['OptionStrike']
    r = x['Risk_free_Rate']
    t = x['Years_to_Expiry']
    mid = x['Mid']
    return P_iv(S, K, r, t, mid)
    
def getting_iv(x):
    option_type = x['OptionType']
    S = x['StockLast']
    K = x['OptionStrike']
    r = x['Risk_free_Rate']
    t = x['Years_to_Expiry']
    mid = x['Mid']
    meth = '{0}_iv'.format(option_type)
    return float(globals().get(meth)(S, K, r, t, mid))
    
    
def func(x):
    return np.exp( -0.5 * x * x)/ (sqrt(2.0 * pi))
def get_c_delta(x):
    S = x['StockLast']
    K = x['OptionStrike']
    r = x['Risk_free_Rate']
    t = x['Years_to_Expiry']
    vol = x['Imp_Vol']
    d1 = (1.0/(vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol**2.0) * t)
    return N(d1)
def get_gamma(x):
    S = x['StockLast']
    K = x['OptionStrike']
    r = x['Risk_free_Rate']
    t = x['Years_to_Expiry']
    vol = x['Imp_Vol']
    d1 = (1.0/(vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol**2.0) * t) 
    return func(d1) / (S * vol * sqrt(t))
def get_vega(x):
    S = x['StockLast']
    K = x['OptionStrike']
    r = x['Risk_free_Rate']
    t = x['Years_to_Expiry']
    vol = x['Imp_Vol']
    d1 = (1.0/(vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol**2.0) * t)
    
    return (S * func(d1) * sqrt(t)) / 100.0
def get_c_rho(x):
    S = x['StockLast']
    K = x['OptionStrike']
    r = x['Risk_free_Rate']
    t = x['Years_to_Expiry']
    vol = x['Imp_Vol']
    d1 = (1.0/(vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol**2.0) * t)
    d2 = d1 - (vol * np.sqrt(t))
    rho = K * t * np.exp(-r * t) * N(d2)
    return rho / 100.0
def get_c_theta(x):
    S = x['StockLast']
    K = x['OptionStrike']
    r = x['Risk_free_Rate']
    t = x['Years_to_Expiry']
    vol = x['Imp_Vol']
    d1 = (1.0/(vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol**2.0) * t)
    d2 = d1 - (vol * np.sqrt(t))
    theta = -((S * func(d1) * vol) / (2.0 * np.sqrt(t))) + (r * K * np.exp(-r * t) * N(-d2))
    return theta / 365.0
def option_moneyness(x):
    S = x['StockLast']
    K = x['OptionStrike']
    return K/S
def log_options_moneyness(x):
    S = x['StockLast']
    K = x['OptionStrike']
    return log(S/K)
def std_moneyness(x):
    S = x['StockLast']
    d = x['Imp_Vol']
    K = x['OptionStrike']
    t = x['Years_to_Expiry']
    m = log(S/K)/(d* sqrt(t))
    return m

#date formatting
def days_till_exp(x):
    exp = x['ExpirationDate']
    date_str = datetime.strptime(exp,'%Y-%m-%d %H:%M:%S')
    time_of_capture = x['TimeStamp']
    time_of_c = datetime.strptime(time_of_capture,'%Y-%m-%d-%H:%M:%S')
    return (date_str - time_of_c).days + 1
def years_till_exp(x):
    exp = x['ExpirationDate']
    date_str = time.strptime(exp,'%Y-%m-%d %H:%M:%S')
    number_of_seconds = time.mktime(date_str)
    time_of_capture = x['TimeStamp']
    time_of_c = datetime.strptime(time_of_capture,'%Y-%m-%d-%H:%M:%S')
    
    seconds_now = time_of_c.timestamp()
    sec_untill_exp = number_of_seconds - seconds_now
    sec_in_year = 31536000.00
    return max(sec_untill_exp / sec_in_year, 1e-10)
duration = [30, 90, 180, 360, 720, 1080, 1800]
rates = [0.0005, 0.0009, 0.001, 0.0011, 0.00112, 0.0012, 0.0025]
def us_interest_rates_for_a_tBill(x):
    days = x['Days_to_Expiry']
    durations = [i for i in range(30, 1801)]
    inter = interp1d(duration, rates, kind = 'linear')
    interp = inter(durations)
    return round(interp[max(days, 30) - 30], 8)
def getting_mid(x):
    bid = x['Bid']
    ask = x['Ask']
    last = x['Last']
    if ask == 0.0 or bid == 0.0:
        return last
    else:
        return (ask + bid)/ 2.0
#implied volatility 
def option_values(x):
    option_type = x['OptionType']
    S = x['StockLast']
    K = x['OptionStrike']
    r = x['Risk_free_Rate']
    t = x['Years_to_Expiry']
    vol = x['Imp_Vol']
    meth = 'bs_{0}_value'.format(option_type)
    return float(globals().get(meth)(S, K, r, t, vol)) 
def error_of_BS(x):
    mid = x['Mid']
    call = x['Option_Value']
    return mid - call

    

options_table['Days_to_Expiry'] = options_table.apply(days_till_exp, axis =1)
options_table['Years_to_Expiry'] = options_table.apply(years_till_exp, axis =1)
options_table['Risk_free_Rate'] = options_table.apply(us_interest_rates_for_a_tBill, axis = 1 )
options_table['Mid'] = options_table.apply(getting_mid, axis = 1)
options_table['Imp_Vol'] = options_table.apply(getting_iv,axis =1)
options_table['Option_Value'] = options_table.apply(option_values, axis = 1)
options_table['Error_of_BS'] = options_table.apply(error_of_BS, axis = 1)
#options_table = options_table.drop(['TimeStamp','LastSize','TickID','BidSize','AskSize','BasisForLast','Position','DaysToExpire','Option','OptionType'], axis = 1)
options_table.head()
options_table.info()
options_table['Delta'] = options_table.apply(get_c_delta, axis =1)
options_table['Vega'] = options_table.apply(get_vega, axis =1)
options_table['Theta'] = options_table.apply(get_c_theta, axis =1)
options_table['Moneyness'] = options_table.apply(option_moneyness, axis = 1)
options_table['Log_Moneyness'] = options_table.apply(log_options_moneyness, axis = 1)
options_table['Gamma'] = options_table.apply(get_gamma, axis =1)
options_table['Rho'] = options_table.apply(get_c_rho, axis =1)
options_table['NormalOp'] = options_table['Option_Value']/options_table['OptionStrike']
options_table.dropna(inplace = True)
options_table.head()
options_table['Std_Moneyness'] = options_table.apply(std_moneyness,axis = 1)
options_table.to_csv('options_table_f.csv')
options_table.to_csv('options_table.csv')
options_table.head()
plt.figure(figsize = (14,10))
plt.hist(options_table['Error_of_BS'],bins = 50, edgecolor = 'black', color = 'white')
plt.xlabel('Error')
plt.ylabel('Density')



plt.figure(figsize = (14,10))
plt.scatter(options_table['Mid'], options_table['Option_Value'],color='black',linewidth=0.5,alpha=0.5, s= 1 )
plt.xlabel('Actual Price',fontsize=20,fontname='Times New Roman')
plt.ylabel('Predicted Price',fontsize=20,fontname='Times New Roman') 
plt.show()
options_table['Days_to_Expiry'].unique()
mon_43 = options_table[options_table['Days_to_Expiry'] == 43 ]
mon_15 = options_table[options_table['Days_to_Expiry'] == 15 ]
mon_78 = options_table[options_table['Days_to_Expiry'] == 78 ]
mon_134 = options_table[options_table['Days_to_Expiry'] == 134 ]
mon_106 = options_table[options_table['Days_to_Expiry'] == 106 ]
mon_470 = options_table[options_table['Days_to_Expiry'] == 470 ]

fig = plt.figure(figsize = (14,10))
ax = plt.axes()
ax.plot(mon_43['Moneyness'], mon_43['NormalOp'], color = 'black', linewidth = 0.8, marker = '*', markersize = 5, alpha = 0.5)
ax.plot(mon_15['Moneyness'], mon_15['NormalOp'], color = 'blue', linewidth = 0.8, marker = 'o', markersize = 5, alpha = 0.5)
ax.plot(mon_78['Moneyness'], mon_78['NormalOp'], color = 'cyan', linewidth = 0.8, marker = 'p', markersize = 5, alpha = 0.5)
ax.plot(mon_134['Moneyness'], mon_134['NormalOp'], color = 'orange', linewidth = 0.8, marker = 'd', markersize = 5, alpha = 0.5)
ax.plot(mon_106['Moneyness'], mon_106['NormalOp'], color = 'brown', linewidth = 0.8, marker = '8', markersize = 5, alpha = 0.5)
ax.plot(mon_470['Moneyness'], mon_470['NormalOp'], color = 'red', linewidth = 0.8, marker = 'P', markersize = 5, alpha = 0.5)

#add plots here
op_43 = options_table[options_table['Days_to_Expiry'] == 15]
plt.scatter(op_43['Imp_Vol'], op_43['OptionStrike'],color='black')


expirations = options_table['Days_to_Expiry'].unique()
iv_multi = options_table[options_table['Days_to_Expiry'].isin(expirations)]
iv_multi.drop_duplicates(inplace = True)
iv_pivoted = iv_multi[['Days_to_Expiry', 'OptionStrike', 'Imp_Vol']].reset_index().pivot_table(index='OptionStrike', columns='Days_to_Expiry', values='Imp_Vol').dropna()
iv_pivoted.plot()
iv_pivoted_surface = iv_multi[['Days_to_Expiry', 'OptionStrike', 'Imp_Vol']].reset_index().pivot_table(index='OptionStrike', columns='Days_to_Expiry', values='Imp_Vol').dropna()
fig = plt.figure()

# add the subplot with projection argument
ax = fig.add_subplot(111, projection='3d')

# get the 1d values from the pivoted dataframe
x, y, z = iv_pivoted_surface.columns.values, iv_pivoted_surface.index.values, iv_pivoted_surface.values

# return coordinate matrices from coordinate vectors
X, Y = np.meshgrid(x, y)
ax.set_xlabel('Days to expiration')
ax.set_ylabel('Strike price')
ax.set_zlabel('Implied volatility')
ax.set_title('Implied volatility surface')

# plot
ax.plot_surface(X, Y, z, rstride=4, cstride=4, color='b')
options_table_pred = options_table
n_options_table = options_table[options_table['Days_to_Expiry'] > 106]
n_options_table.head(1)

n_options_table = options_table[options_table['Days_to_Expiry'] >= 78]

from scipy.interpolate import griddata
import plotly.graph_objects as go
x = n_options_table['Years_to_Expiry']
y = n_options_table['Moneyness']
z = n_options_table['Imp_Vol']
x = np.array(x)
y = np.array(y)
z = np.array(z)
# x = np.reshape(x, (2, 1136))
# y = np.reshape(y, (2, 1136))
# z = np.reshape(z, (2, 1136))

# fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
# fig.update_layout(title='Mt Bruno Elevation', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
# fig.show()
# plt.figure(figsize = (14,10))
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none' )
# ax.set_xlabel('Years to Expiry', fontsize=20)
# ax.set_ylabel('Moneyness', fontsize = 20)
# ax.set_zlabel('Implied Volatility', fontsize=20, rotation = 0)
def make_surf(X,Y,Z):
    XX,YY = np.meshgrid(np.linspace(min(X),max(X),500),np.linspace(min(Y),max(Y),1000))
    ZZ = griddata(np.array([X,Y]).T,np.array(Z),(XX,YY), method='linear')
    return XX,YY,ZZ
make_surf(x,y,z)

def mesh_plot2(X,Y,Z):
    fig = plt.figure()
    ax = Axes3D(fig, azim = -40, elev = 60)
    XX,YY,ZZ = make_surf(X,Y,Z)

    #fig = go.Figure(data=[go.Surface(x=XX, y=YY, z=ZZ)])
    fig = ax.plot_surface(XX, YY, ZZ, rstride=4, cstride=4, color='b')
    return fig

mesh_plot2(x,y,z)

options_table['StockLast']
options_table.info()
options_table.dropna(inplace = True)
options_table['StockLast'] = options_table['StockLast']/ options_table['OptionStrike']
options_table['Option_Value'] = options_table['Option_Value']/ options_table['OptionStrike']


options_table.info()
n = len(options_table)
n_t = (int)(0.8 * n)

train = options_table[0:n_t]
X_train = train[['StockLast', 'Years_to_Expiry', 'Risk_free_Rate','Imp_Vol']].values
y_train = train[['Option_Value']].values
test = options_table[n_t+1:n]
X_test = test[['StockLast', 'Years_to_Expiry', 'Risk_free_Rate','Imp_Vol']].values
y_test = test[['Option_Value']].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
options_table['StockLast']
y_train
X_train.shape[1]
X_train
def custom_activation(x):
    return backend.exp(x)

nodes = 120
model = Sequential()

model.add(Dense(nodes, input_dim=X_train.shape[1], kernel_initializer = 'glorot_uniform'))
model.add(LeakyReLU())
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='elu'))
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='elu'))
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='elu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation(custom_activation))
          
model.compile(loss='mse',optimizer='adam')
history = model.fit(X_train, y_train, batch_size=128, epochs=500, validation_split=0.3, verbose= 2)
model.save('options_500.h5')
print(history.history.keys())
plt.figure(figsize = (14,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.title('Model Loss(Training)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
predictions = model.predict(X_train)
y_train
np.count_nonzero(~np.isnan(predictions))

predictions.flatten()
y_train

plt.figure(figsize=(14,10))
plt.scatter(y_train , predictions, color='black',linewidth=0.3,alpha = 0.4, s = 0.9)

plt.xlabel('Actual Price',fontsize=20)
plt.ylabel('Predicted Price',fontsize=20)



plt.figure(figsize=(14,10))
plt.scatter(y_train , predictions, color='black',linewidth=0.3,alpha = 0.4, s = 0.9)
plt.xlim(0, 0.25)
plt.ylim(0, 0.25)

plt.xlabel('Actual Price',fontsize=20)
plt.ylabel('Predicted Price',fontsize=20)

predictions = predictions.flatten()

y_train = y_train.flatten()
q = pd.DataFrame()
q['difference'] = y_train - predictions
q['mse'] = mean(q['difference']**2)
print("Mean Squared Error:  ",q['mse'].mean())
q['mae'] = mean(abs(q['difference']))
print("Mean Absolute Error:  ",q['mae'].mean())
y_test = y_test.flatten()
y_test_final = model.predict(X_test)
y_test_final = y_test_final.flatten()
t = pd.DataFrame()
t['difference'] = y_test - y_test_final
t['mse'] = mean(t['difference']**2)
print("Mean Squared Error:  ",t['mse'].mean())
t['mae'] = mean(abs(t['difference']))
print("Mean Absolute Error:  ",t['mae'].mean())

plt.figure(figsize=(14,10))
plt.scatter(y_test, y_test_final, color='black',linewidth=0.5,alpha = 0.4, s = 1)
plt.xlabel('Actual Price',fontsize=20)

plt.ylabel('Predicted Price',fontsize=20)

plt.xlim(0, 0.25)
plt.ylim(0, 0.25)
plt.figure(figsize=(14,10))
plt.hist(y_test - y_test_final, bins=50,edgecolor='black',color='white')
plt.xlabel('Error')
plt.ylabel('Density')
plt.show()
newop = options_table.copy()
options_table_pre = options_table[['StockLast', 'Years_to_Expiry', 'Risk_free_Rate','Imp_Vol']]
options_table.head()
options_table_pre['Actual_Stock_last'] = options_table_pre['StockLast']* options_table['OptionStrike']

options_table_pre
options_table['PredictedOP'] = model.predict(options_table[['StockLast', 'Years_to_Expiry', 'Risk_free_Rate','Imp_Vol']])
options_table_pred
options_table['Normal_op_act'] = options_table['NormalOp']* options_table['OptionStrike']
options_table['Predicted_op_act'] = options_table['PredictedOP']* options_table['OptionStrike']

days = options_table['Days_to_Expiry'].unique()
days
pred43 = options_table[options_table['Days_to_Expiry'] == 10]
plt.figure(figsize = (14,10))
ax = plt.axes()
ax.plot(pred43['Normal_op_act'])
ax.plot(pred43['Predicted_op_act'])
model.save('options_100.h5')
#heston modelling

strike_price = 110.0
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
# option data
maturity_date = ql.Date(15, 1, 2016)
spot_price = 127.62
strike_price = 130
volatility = 0.20 # the historical vols for a year
dividend_rate =  0
option_type = ql.Option.Call

risk_free_rate = 0.001
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()

calculation_date = ql.Date(8, 5, 2015)
ql.Settings.instance().evaluationDate = calculation_date

payoff = ql.PlainVanillaPayoff(option_type, strike_price)
exercise = ql.EuropeanExercise(maturity_date)
european_option = ql.VanillaOption(payoff, exercise)

v0 = volatility*volatility  # spot variance
kappa = 0.1
theta = v0
sigma = 0.1
rho = -0.75

spot_handle = ql.QuoteHandle(
    ql.SimpleQuote(spot_price)
)
flat_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, day_count)
)
dividend_yield = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, dividend_rate, day_count)
)
heston_process = ql.HestonProcess(flat_ts,
                                  dividend_yield,
                                  spot_handle,
                                  v0,
                                  kappa,
                                  theta,
                                  sigma,
                                  rho)
engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.01, 1000)
european_option.setPricingEngine(engine)
h_price = european_option.NPV()
print("The Heston model price is",h_price)
print(spot_handle)

