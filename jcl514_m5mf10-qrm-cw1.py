import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2
from arch.univariate import arch_model
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
df_A = pd.read_csv('../input/m5mf10-qrm-cw-j-li/QRM-2018-cw1-data-1.csv')
df_A.head()
df_A.shape
log_returns_A = 100*(np.array(np.log(df_A.iloc[1:,1])) - np.array(np.log(df_A.iloc[:-1,1])))
# Based on definitions from lecture slides
mean_A = np.mean(log_returns_A)
sd_A = np.std(log_returns_A,ddof = 1)
skew_A = np.mean((log_returns_A - mean_A)**3)/(np.std(log_returns_A,ddof=0)**3)
kurt_A = np.mean((log_returns_A - mean_A)**4)/(np.std(log_returns_A,ddof=0)**4)
print('Mean: ', mean_A)
print('Standard Deviation: ', sd_A)
print('Skewness: ', skew_A)
print('Kurtosis: ', kurt_A)
def kde(data, plotrange):
    
    T = data.shape[0]
    h = 1.06 * np.std(data, ddof = 1) * (T**(-0.2))
    R = np.array([list(data),]*plotrange.shape[0]).T
    X = np.array([list(plotrange),]*T)
    kde = np.mean(norm.pdf((R-X)/h), axis=0)/h
    
    return kde

# Define Plot Ranges
x = np.arange(min(log_returns_A),max(log_returns_A),0.1)
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('Distributions of Log Return per Barrel of WTI')
plt.xlabel('Log Return (%)')
plt.ylabel('Density')
# Create Legends
histogram_legend = Rectangle((0,0),1,1,color='lightblue', label = 'Histogram')
kernel_legend = mlines.Line2D([],[],color='blue', label='Kernel')
normal_legend = mlines.Line2D([],[],color='red', label='Normal')
plt.legend(handles=[histogram_legend, kernel_legend, normal_legend])
# Create Plots
plot1 = plt.hist(log_returns_A, bins = 'auto', density = True, color = 'lightblue')
plot2 = plt.plot(x,kde(log_returns_A, x), color = 'blue', linestyle = '--')
plot3 = plt.plot(x,norm(loc = mean_A, scale = sd_A).pdf(x), color='red')
plt.savefig('aiplot1.jpg')
def acf(data): # input should be 1-d numpy array
    T = data.shape[0]
    # Center Data
    data_c = data - np.mean(data)
    # Calculate numerators of rhos for each lag 
    rho = np.array([sum(data_c[:T-h]*data_c[h:]) for h in np.arange(T)])
    # Calculate empirical ACFs for each lag
    rho = rho/sum(data_c**2)
    return rho
acf_A = acf(log_returns_A)
acf_absA = acf(abs(log_returns_A))
acf_A2 = acf(log_returns_A**2)
# Define Plot Ranges
x = np.arange(30)
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('ACFs of Log Return per Barrel of WTI')
plt.xlabel('Lag')
plt.ylabel('ACF')
# Create Plots
plot1 = plt.bar(x, acf_A[:30], width = 0.8 ,color = 'orange')
plt.savefig('aiiplot1.jpg')
# Define Plot Ranges
x = np.arange(30)
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('ACFs of Absolute Log Return per Barrel of WTI')
plt.xlabel('Lag')
plt.ylabel('ACF')
# Create Plots
plot2 = plt.bar(x, acf_absA[:30], width = 0.8 , color = 'blue')
plt.savefig('aiiplot2.jpg')
# Define Plot Ranges
x = np.arange(30)
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('ACFs of Squared Log Return per Barrel of WTI')
plt.xlabel('Lag')
plt.ylabel('ACF')
# Create Plots
plot3 = plt.bar(x, acf_A2[:30], width = 0.8 , color='red')
plt.savefig('aiiplot3.jpg')
M1 = arch_model(log_returns_A, mean='Constant', vol='GARCH', p=1, q=1, dist = 'gaussian')
fit1 = M1.fit(disp='off', cov_type='classic')
print('Fitted Parameter Values')
print(fit1.params)
print('Standard Errors of Fitted Parameter Values')
print(fit1.std_err)
# Define Plot Ranges
x = pd.DataFrame()
x['temp'] = abs(log_returns_A - fit1.params['mu'])
x = x.set_index(pd.to_datetime(df_A.iloc[1:,0]))
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('Plot of Absolute Centered Log Returns and Fitted Volatilities against Time')
plt.xlabel('Time')
plt.ylabel('|Log Return| (%)')
# Create Legends
absA_legend = mlines.Line2D([],[],color='orange', label='Absolute Log Return')
vol_legend = mlines.Line2D([],[],color='blue', label='Fitted Volatility')
plt.legend(handles=[absA_legend, vol_legend])
# Create Plots
plot1 = plt.plot(x, color = 'orange')
x['temp'] = fit1.conditional_volatility
plot2 = plt.plot(x, color='blue')
plt.savefig('aiiiplot1.jpg')
res_A = (log_returns_A - fit1.params['mu'])/fit1.conditional_volatility
# Define Plot Ranges
x = np.arange(min(res_A),max(res_A),0.1)
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('Distributions of Standardized Residuals of GARCH(1,1) with Standard Normal Errors')
plt.xlabel('Standardized Residuals')
plt.ylabel('Density')
# Create Legends
histogram_legend = Rectangle((0,0),1,1,color='lightblue', label = 'Histogram')
kernel_legend = mlines.Line2D([],[],color='blue', label='Kernel')
normal_legend = mlines.Line2D([],[],color='red', label='Normal')
plt.legend(handles=[histogram_legend, kernel_legend, normal_legend])
# Create Plots
plot1 = plt.hist(res_A, bins = 'auto', density = True, color = 'lightblue')
plot2 = plt.plot(x,kde(res_A, x), color = 'blue', linestyle = '--')
#plot3 = plt.plot(x,norm(loc = np.mean(res_A), scale = np.std(res_A,ddof = 1)).pdf(x), color='red')
plot3 = plt.plot(x,norm(loc = 0, scale = 1).pdf(x), color='red')
plt.savefig('aivplot1.jpg')
plot4 = stats.probplot(res_A,dist='norm',fit=True,plot=plt)
plt.ylabel('Sample Quantiles')
plt.xlabel('Theoretical Quantiles')
plt.title('Normal Q-Q Plot of Standardized Residuals')
plt.savefig('aivplot2.jpg')
# Define Plot Ranges
x = np.arange(30)
acf_res_A = acf(res_A)
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('ACFs of Standardized Residuals of GARCH(1,1) with Standard Normal Errors')
plt.xlabel('Lag')
plt.ylabel('ACF')
# Create Plots
plot1 = plt.bar(x, acf_res_A[:30], width = 0.8 ,color = 'orange')
plt.savefig('aivplot3.jpg')
# Define Plot Ranges
x = np.arange(30)
acf_absres_A = acf(abs(res_A))
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('ACFs of Absolute Standardized Residuals of GARCH(1,1) with Standard Normal Errors')
plt.xlabel('Lag')
plt.ylabel('ACF')
# Create Plots
plot1 = plt.bar(x, acf_absres_A[:30], width = 0.8 ,color = 'blue')
plt.savefig('aivplot4.jpg')
# From R

# Mean Model fitted parameter values: mu = 0.039037, ar1 = -0.835566, ma1 = 0.822915
# Volatility Model fitted parameter values: omega = 0.029835, alpha1 = 0.062039, beta1 = 0.932835
# t-distribution fitted shape parameter: 7.203823

# Mean Model fitted parameter std errors: mu = 0.032563, ar1 = 0.337018, ma1 = 0.348743
# Volatility Model fitted parameter std errors: omega = 0.012674, alpha1 = 0.010984, beta1 = 0.011514
# t-distribution fitted shape parameter std errors: 0.948127
# Note that the aboves are non-robust errors
muhat_q2v = pd.read_csv('../input/q2vdata/muhat.csv')
res_q2v = pd.read_csv('../input/q2vdata/res.csv')
sigmahat_q2v = pd.read_csv('../input/q2vdata/sigmahat.csv')
muhat_q2v = np.array(muhat_q2v.iloc[:,1])
res_q2v = np.array(res_q2v.iloc[:,1])
sigmahat_q2v = np.array(sigmahat_q2v.iloc[:,1])
# Define Plot Ranges
x = pd.DataFrame()
x['temp'] = abs(log_returns_A - muhat_q2v)
x = x.set_index(pd.to_datetime(df_A.iloc[1:,0]))
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('Plot of Absolute Centered Log Returns and Fitted Volatilities against Time')
plt.xlabel('Time')
plt.ylabel('|Log Return| (%)')
# Create Legends
absA_legend = mlines.Line2D([],[],color='orange', label='Absolute Log Return')
vol_legend = mlines.Line2D([],[],color='blue', label='Fitted Volatility')
plt.legend(handles=[absA_legend, vol_legend])
# Create Plots
plot1 = plt.plot(x, color = 'orange')
x['temp'] = sigmahat_q2v
plot2 = plt.plot(x, color='blue')
plt.savefig('avplot1.jpg')
# Define Plot Ranges
x = np.arange(min(res_q2v),max(res_q2v),0.1)
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('Distributions of Standardized Residuals of ARCH(1,1)-GARCH(1,1) with Student t Errors')
plt.xlabel('Standardized Residuals')
plt.ylabel('Density')
# Create Legends
histogram_legend = Rectangle((0,0),1,1,color='lightblue', label = 'Histogram')
kernel_legend = mlines.Line2D([],[],color='blue', label='Kernel')
normal_legend = mlines.Line2D([],[],color='red', label='Student t')
plt.legend(handles=[histogram_legend, kernel_legend, normal_legend])
# Create Plots
plot1 = plt.hist(res_q2v, bins = 'auto', density = True, color = 'lightblue')
plot2 = plt.plot(x,kde(res_q2v, x), color = 'blue', linestyle = '--')
#plot3 = plt.plot(x,norm(loc = np.mean(res_A), scale = np.std(res_A,ddof = 1)).pdf(x), color='red')
plot3 = plt.plot(x,t(scale = np.sqrt(5.203823/7.203823), df = 7.203823).pdf(x), color='red')
plt.savefig('avplot2.jpg')
plot4 = stats.probplot(res_q2v,dist=stats.t(scale = np.sqrt(5.203823/7.203823), df = 7.203823),fit=True,plot=plt)
plt.ylabel('Sample Quantiles')
plt.xlabel('Theoretical Quantiles')
plt.title('Student t Q-Q Plot of Standardized Residuals')
plt.savefig('avplot3.jpg')
# Define Plot Ranges
x = np.arange(30)
acf_res_q2v = acf(res_q2v)
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('ACFs of Standardized Residuals of ARCH(1,1)-GARCH(1,1) with Student t Errors')
plt.xlabel('Lag')
plt.ylabel('ACF')
# Create Plots
plot1 = plt.bar(x, acf_res_q2v[:30], width = 0.8 ,color = 'orange')
plt.savefig('avplot4.jpg')
# Define Plot Ranges
x = np.arange(30)
acf_absres_q2v = acf(abs(res_q2v))
# Define Plot Size
plt.figure(figsize=(10,5))
# Create Plot and Axes Titles
plt.title('ACFs of Absolute Standardized Residuals of ARCH(1,1)-GARCH(1,1) with Student t Errors')
plt.xlabel('Lag')
plt.ylabel('ACF')
# Create Plots
plot1 = plt.bar(x, acf_absres_q2v[:30], width = 0.8 ,color = 'blue')
plt.savefig('avplot5.jpg')
df_B = pd.read_csv('../input/m5mf10-qrm-cw-j-li/QRM-2018-cw1-data-2.csv')
df_B.head()
loss_B = -100*(np.array(np.log(df_B.iloc[1:,1])) - np.array(np.log(df_B.iloc[:-1,1])))
# 504th entry (python index 503) of df_B is 10/11/2008 
df_B.iloc[503,:]
# Hence the linearized loss on 10/11/2008 is the 503th entry (python index 502) of loss_B.
# It means that the the first loss distribution should be estimated upto 502th entry (python 
# index 501)
loss_B[501]
def EVaR(data, alpha):
    n = data.shape[0]
    data_ = np.sort(data)
    return data_[int(np.floor(alpha*n))]
VaR_fore_1_95 = np.array([EVaR(loss_B[2+k:502+k], 0.95) for k in np.arange(2519)])
VaR_fore_1_99 = np.array([EVaR(loss_B[2+k:502+k], 0.99) for k in np.arange(2519)])
def EES(data,alpha):
    data_ = data[np.where(data >= EVaR(data, alpha))]
    return np.mean(data_)
ES_fore_1_95 = np.array([EES(loss_B[2+k:502+k], 0.95) for k in np.arange(2519)])
ES_fore_1_99 = np.array([EES(loss_B[2+k:502+k], 0.99) for k in np.arange(2519)])
EWMA_fore = np.ones(loss_B.shape[0]+1)
for k in np.arange(loss_B.shape[0]):
    EWMA_fore[k+1] = np.sqrt(0.06*(loss_B[k]**2) + 0.94*(EWMA_fore[k]**2))
EWMA_fore = EWMA_fore[1:]
Z = loss_B/EWMA_fore
VaR_fore_2_95 = np.array([EWMA_fore[502+k] * EVaR(Z[2+k:502+k], 0.95) for k in np.arange(2519)])
VaR_fore_2_99 = np.array([EWMA_fore[502+k] * EVaR(Z[2+k:502+k], 0.99) for k in np.arange(2519)])
ES_fore_2_95 = np.array([EWMA_fore[502+k] * EES(Z[2+k:502+k], 0.95) for k in np.arange(2519)])
ES_fore_2_99 = np.array([EWMA_fore[502+k] * EES(Z[2+k:502+k], 0.99) for k in np.arange(2519)])
VaR_M3_95 = np.zeros(2519)
VaR_M3_99 = np.zeros(2519)
ES_M3_95 = np.zeros(2519)
ES_M3_99 = np.zeros(2519) 
muhat_M3 = np.zeros(2519) # stores one step mean forecast
sigmahat_M3 = np.zeros(2519) # stores one step volatility forecast

for k in np.arange(2519):
    if (k%100 ==0):
        print ('Iteration ' + str(k+1))
    M3 = arch_model(loss_B[2+k:502+k], mean='Constant', vol='GARCH', p=1, q=1, dist = 'gaussian')
    fit3 = M3.fit(disp='off')    
    mu = fit3.params['mu']
    ome = fit3.params['omega']
    alp = fit3.params['alpha[1]']
    bet = fit3.params['beta[1]']
    sig = fit3.conditional_volatility[499]
    muhat_M3[k] = mu
    sigmahat_M3[k] = np.sqrt(ome + alp * ((loss_B[501+k] - mu)**2) + bet*(sig**2))
    Z_M3 = (loss_B[2+k:502+k] - mu)/fit3.conditional_volatility
    VaR_M3_95[k] = EVaR(Z_M3, 0.95)
    VaR_M3_99[k] = EVaR(Z_M3, 0.99)
    ES_M3_95[k] = EES(Z_M3,0.95)
    ES_M3_99[k] = EES(Z_M3, 0.99)
VaR_fore_3_95 = muhat_M3 + sigmahat_M3*VaR_M3_95
VaR_fore_3_99 = muhat_M3 + sigmahat_M3*VaR_M3_99
ES_fore_3_95 = muhat_M3 + sigmahat_M3*ES_M3_95
ES_fore_3_99 = muhat_M3 + sigmahat_M3*ES_M3_99
# Define Plot Ranges
x = pd.DataFrame()
x['temp'] = np.array([max(0,k) for k in loss_B[502:]])
x = x.set_index(pd.to_datetime(df_B.iloc[503:,0]))
# Define Plot Size
plt.figure(figsize=(12,5))
# Create Plot and Axes Titles
plt.title('Plot of Positive Part of Actual Losses with VaR and ES Forecasts using HS against Time')
plt.xlabel('Time')
plt.ylabel('Postive Part of Daily Linearized Losses (%)')
# Create Legends
loss_legend = mlines.Line2D([],[],color='orange', label='Actual Losses')
ES_95_legend = mlines.Line2D([],[],color='red', label='95% ES')
VaR_95_legend = mlines.Line2D([],[],color='red', linestyle = '--', label='95% VaR')
ES_99_legend = mlines.Line2D([],[],color='blue', label='99% ES')
VaR_99_legend = mlines.Line2D([],[],color='blue', linestyle = '--', label='99% VaR')
plt.legend(handles=[loss_legend, ES_95_legend, VaR_95_legend, ES_99_legend, VaR_99_legend])
# Fix y limit for comparison purpose
plt.ylim(top=40)
# Create Plots
plot1 = plt.plot(x, color = 'orange', linewidth = 1)
x['temp'] = VaR_fore_1_95
plot2 = plt.plot(x, color='red', linestyle='--', linewidth = 1)
x['temp'] = VaR_fore_1_99
plot3 = plt.plot(x, color='blue', linestyle='--', linewidth = 1)
x['temp'] = ES_fore_1_95
plot4 = plt.plot(x, color='red', linewidth = 1)
x['temp'] = ES_fore_1_99
plot5 = plt.plot(x, color='blue', linewidth = 1)
plt.savefig('biplot1.jpg')
# Define Plot Ranges
x = pd.DataFrame()
x['temp'] = np.array([max(0,k) for k in loss_B[502:]])
x = x.set_index(pd.to_datetime(df_B.iloc[503:,0]))
# Define Plot Size
plt.figure(figsize=(13,5))
# Create Plot and Axes Titles
plt.title('Plot of Positive Part of Actual Losses with 99% VaR Forecasts and Violations using HS against Time')
plt.xlabel('Time')
plt.ylabel('Postive Part of Daily Linearized Losses (%)')
# Create Legends
loss_legend = mlines.Line2D([],[],color='orange', label='Actual Losses')
VaR_99_legend = mlines.Line2D([],[],color='blue', linestyle = '--', label='99% VaR')
vio_legend = mlines.Line2D([],[],color='white', marker = 'o', markerfacecolor = 'red', label='Violations', markersize=8)
plt.legend(handles=[loss_legend, VaR_99_legend, vio_legend])
# Fix y limit for comparison purpose
plt.ylim(top=40)
# Create Plots
plot1 = plt.plot(x, color = 'orange', linewidth = 1, zorder = -1)
x['temp'] = VaR_fore_1_99
plot3 = plt.plot(x, color = 'blue', linestyle = '--', linewidth = 1, zorder = -1)
x['temp'] = loss_B[502:]
index = np.where((loss_B[502:]>VaR_fore_1_99) > 0)[0]
dates = [pd.to_datetime(d) for d in df_B.iloc[503:,0].iloc[index]]
plot5 = plt.scatter(dates, x.iloc[index], color='red', marker = 'o', zorder = 1, s=23)
plt.savefig('biplot2.jpg')
# Define Plot Ranges
x = pd.DataFrame()
x['temp'] = np.array([max(0,k) for k in loss_B[502:]])
x = x.set_index(pd.to_datetime(df_B.iloc[503:,0]))
# Define Plot Size
plt.figure(figsize=(12,5))
# Create Plot and Axes Titles
plt.title('Plot of Positive Part of Actual Losses with VaR and ES Forecasts using FHS with EWMA against Time')
plt.xlabel('Time')
plt.ylabel('Postive Part of Daily Linearized Losses (%)')
# Create Legends
loss_legend = mlines.Line2D([],[],color='orange', label='Actual Losses')
ES_95_legend = mlines.Line2D([],[],color='red', label='95% ES')
VaR_95_legend = mlines.Line2D([],[],color='red', linestyle = '--', label='95% VaR')
ES_99_legend = mlines.Line2D([],[],color='blue', label='99% ES')
VaR_99_legend = mlines.Line2D([],[],color='blue', linestyle = '--', label='99% VaR')
plt.legend(handles=[loss_legend, ES_95_legend, VaR_95_legend, ES_99_legend, VaR_99_legend])
# Fix y limit for comparison purpose
plt.ylim(top=40)
# Create Plots
plot1 = plt.plot(x, color = 'orange', linewidth = 1)
x['temp'] = VaR_fore_2_95
plot2 = plt.plot(x, color='red', linestyle='--', linewidth = 1)
x['temp'] = VaR_fore_2_99
plot3 = plt.plot(x, color='blue', linestyle='--', linewidth = 1)
x['temp'] = ES_fore_2_95
plot4 = plt.plot(x, color='red', linewidth = 1)
x['temp'] = ES_fore_2_99
plot5 = plt.plot(x, color='blue', linewidth = 1)
plt.savefig('biiplot1.jpg')
# Define Plot Ranges
x = pd.DataFrame()
x['temp'] = np.array([max(0,k) for k in loss_B[502:]])
x = x.set_index(pd.to_datetime(df_B.iloc[503:,0]))
# Define Plot Size
plt.figure(figsize=(13,5))
# Create Plot and Axes Titles
plt.title('Plot of Positive Part of Actual Losses with 99% VaR Forecasts and Violations using FHS with EWMA against Time')
plt.xlabel('Time')
plt.ylabel('Postive Part of Daily Linearized Losses (%)')
# Create Legends
loss_legend = mlines.Line2D([],[],color='orange', label='Actual Losses')
VaR_99_legend = mlines.Line2D([],[],color='blue', linestyle = '--', label='99% VaR')
vio_legend = mlines.Line2D([],[],color='white', marker = 'o', markerfacecolor = 'red', label='Violations', markersize=8)
plt.legend(handles=[loss_legend, VaR_99_legend, vio_legend])
# Fix y limit for comparison purpose
plt.ylim(top=40)
# Create Plots
plot1 = plt.plot(x, color = 'orange', linewidth = 1, zorder = -1)
x['temp'] = VaR_fore_2_99
plot3 = plt.plot(x, color = 'blue', linestyle = '--', linewidth = 1, zorder = -1)
x['temp'] = loss_B[502:]
index = np.where((loss_B[502:]>VaR_fore_2_99) > 0)[0]
dates = [pd.to_datetime(d) for d in df_B.iloc[503:,0].iloc[index]]
plot5 = plt.scatter(dates, x.iloc[index], color='red', marker = 'o', zorder = 1, s=23)
plt.savefig('biiplot2.jpg')
# Define Plot Ranges
x = pd.DataFrame()
x['temp'] = np.array([max(0,k) for k in loss_B[502:]])
x = x.set_index(pd.to_datetime(df_B.iloc[503:,0]))
# Define Plot Size
plt.figure(figsize=(12,5))
# Create Plot and Axes Titles
plt.title('Plot of Positive Part of Actual Losses with VaR and ES Forecasts using FHS with GARCH(1,1) against Time')
plt.xlabel('Time')
plt.ylabel('Postive Part of Daily Linearized Losses (%)')
# Create Legends
loss_legend = mlines.Line2D([],[],color='orange', label='Actual Losses')
ES_95_legend = mlines.Line2D([],[],color='red', label='95% ES')
VaR_95_legend = mlines.Line2D([],[],color='red', linestyle = '--', label='95% VaR')
ES_99_legend = mlines.Line2D([],[],color='blue', label='99% ES')
VaR_99_legend = mlines.Line2D([],[],color='blue', linestyle = '--', label='99% VaR')
plt.legend(handles=[loss_legend, ES_95_legend, VaR_95_legend, ES_99_legend, VaR_99_legend])
# Fix y limit for comparison purpose
plt.ylim(top=40)
# Create Plots
plot1 = plt.plot(x, color = 'orange', linewidth = 1)
x['temp'] = VaR_fore_3_95
plot2 = plt.plot(x, color='red', linestyle='--', linewidth = 1)
x['temp'] = VaR_fore_3_99
plot3 = plt.plot(x, color='blue', linestyle='--', linewidth = 1)
x['temp'] = ES_fore_3_95
plot4 = plt.plot(x, color='red', linewidth = 1)
x['temp'] = ES_fore_3_99
plot5 = plt.plot(x, color='blue', linewidth = 1)
plt.savefig('biiiplot1.jpg')
# Define Plot Ranges
x = pd.DataFrame()
x['temp'] = np.array([max(0,k) for k in loss_B[502:]])
x = x.set_index(pd.to_datetime(df_B.iloc[503:,0]))
# Define Plot Size
plt.figure(figsize=(13,5))
# Create Plot and Axes Titles
plt.title('Plot of Positive Part of Actual Losses with 99% VaR Forecasts and Violations using FHS with GARCH(1,1) against Time')
plt.xlabel('Time')
plt.ylabel('Postive Part of Daily Linearized Losses (%)')
# Create Legends
loss_legend = mlines.Line2D([],[],color='orange', label='Actual Losses')
VaR_99_legend = mlines.Line2D([],[],color='blue', linestyle = '--', label='99% VaR')
vio_legend = mlines.Line2D([],[],color='white', marker = 'o', markerfacecolor = 'red', label='Violations', markersize=8)
plt.legend(handles=[loss_legend, VaR_99_legend, vio_legend])
# Fix y limit for comparison purpose
plt.ylim(top=40)
# Create Plots
plot1 = plt.plot(x, color = 'orange', zorder=-1, linewidth = 1)
x['temp'] = VaR_fore_3_99
plot3 = plt.plot(x, color = 'blue', linestyle = '--', zorder=-1, linewidth = 1)
x['temp'] = loss_B[502:]
index = np.where((loss_B[502:]>VaR_fore_3_99) > 0)[0]
dates = [pd.to_datetime(d) for d in df_B.iloc[503:,0].iloc[index]]
plot5 = plt.scatter(dates, x.iloc[index], color='red', marker = 'o', zorder=1, s=23)
plt.savefig('biiiplot2.jpg')
def LRuc(loss,VaR,alpha):
    temp = loss > VaR
    pihat = np.mean(temp)
    vio = np.sum(temp)
    LRuc = -2*np.log(((alpha/(1-pihat))**(temp.shape[0]-vio))*(((1-alpha)/pihat)**vio))
    #LRuc = np.prod((alpha**(1-temp))*((1-alpha)**temp))/np.prod(((1-pihat)**(1-temp))*(pihat**temp))
    #LRuc = -2*np.log(LRuc)
    p = 1 - chi2.cdf(LRuc, df=1)
    return [vio, LRuc, p]
[vio_1_95, LRuc_1_95, puc_1_95] = LRuc(loss_B[502:], VaR_fore_1_95, 0.95)
[vio_2_95, LRuc_2_95, puc_2_95] = LRuc(loss_B[502:], VaR_fore_2_95, 0.95)
[vio_3_95, LRuc_3_95, puc_3_95] = LRuc(loss_B[502:], VaR_fore_3_95, 0.95)
[vio_1_99, LRuc_1_99, puc_1_99] = LRuc(loss_B[502:], VaR_fore_1_99, 0.99)
[vio_2_99, LRuc_2_99, puc_2_99] = LRuc(loss_B[502:], VaR_fore_2_99, 0.99)
[vio_3_99, LRuc_3_99, puc_3_99] = LRuc(loss_B[502:], VaR_fore_3_99, 0.99)
table_UC = pd.DataFrame()
table_UC['VaR Forecast Methods'] = ['HS','FHS with EWMA', 'FHS with GARCH']*2
table_UC['Test Significance Level'] = ['95%']*3 + ['99%']*3
table_UC['Violations'] = [vio_1_95, vio_2_95, vio_3_95, vio_1_99, vio_2_99, vio_3_99]
table_UC['Expected Violations'] = [round(0.05*2519)]*3 + [round(0.01*2519)]*3
table_UC['LRuc'] = [LRuc_1_95, LRuc_2_95, LRuc_3_95, LRuc_1_99, LRuc_2_99, LRuc_3_99]
table_UC['p'] = [puc_1_95, puc_2_95, puc_3_95, puc_1_99, puc_2_99, puc_3_99]
table_UC
def LRjci(loss,VaR,alpha):
    temp = loss > VaR
    T00 = np.sum([1 if (np.sum(temp[k:k+2]) == 0) else 0 for k in np.arange(temp.shape[0] - 1)])
    T11 = np.sum([1 if (np.sum(temp[k:k+2]) == 2) else 0 for k in np.arange(temp.shape[0] - 1)])
    T01 = np.sum([1 if (int(temp[k+1])-int(temp[k])== 1) else 0 for k in np.arange(temp.shape[0] - 1)])
    T10 = np.sum([1 if (int(temp[k+1])-int(temp[k])== -1) else 0 for k in np.arange(temp.shape[0] - 1)])
    pi11 = T11/(T10 + T11)
    pi01 = T01/(T00 + T01)
    pi = np.mean(temp)
    LRind = ((1-pi)**(T00+T10))*(pi**(T01+T11))
    LRind = -2*np.log(LRind/(((1-pi01)**T00) * (pi01**T01) * ((1-pi11)**T10) * (pi11**T11)))
    LRjci = LRind + LRuc(loss,VaR,alpha)[1]
    p = 1 - chi2.cdf(LRjci, df=2)
    return [LRjci, p]
[LRjci_1_95, pjci_1_95] = LRjci(loss_B[502:], VaR_fore_1_95, 0.95)
[LRjci_2_95, pjci_2_95] = LRjci(loss_B[502:], VaR_fore_2_95, 0.95)
[LRjci_3_95, pjci_3_95] = LRjci(loss_B[502:], VaR_fore_3_95, 0.95)
[LRjci_1_99, pjci_1_99] = LRjci(loss_B[502:], VaR_fore_1_99, 0.99)
[LRjci_2_99, pjci_2_99] = LRjci(loss_B[502:], VaR_fore_2_99, 0.99)
[LRjci_3_99, pjci_3_99] = LRjci(loss_B[502:], VaR_fore_3_99, 0.99)
table_jci = pd.DataFrame()
table_jci['VaR Forecast Methods'] = ['HS','FHS with EWMA', 'FHS with GARCH']*2
table_jci['Test Significance Level'] = ['95%']*3 + ['99%']*3
table_jci['LRuc'] = [LRjci_1_95, LRjci_2_95, LRjci_3_95, LRjci_1_99, LRjci_2_99, LRjci_3_99]
table_jci['p'] = [pjci_1_95, pjci_2_95, pjci_3_95, pjci_1_99, pjci_2_99, pjci_3_99]
table_jci
def btES(loss,VaR,ES):
    temp = loss > VaR
    xi = (loss - ES)*temp
    Z = np.sum(xi)/np.sqrt(np.sum(xi**2))
    p = 1 - norm.cdf(Z)
    return [Z, p]
[btES_1_95, pES_1_95] = btES(loss_B[502:], VaR_fore_1_95, ES_fore_1_95)
[btES_2_95, pES_2_95] = btES(loss_B[502:], VaR_fore_2_95, ES_fore_2_95)
[btES_3_95, pES_3_95] = btES(loss_B[502:], VaR_fore_3_95, ES_fore_3_95)
[btES_1_99, pES_1_99] = btES(loss_B[502:], VaR_fore_1_99, ES_fore_1_99)
[btES_2_99, pES_2_99] = btES(loss_B[502:], VaR_fore_2_99, ES_fore_2_99)
[btES_3_99, pES_3_99] = btES(loss_B[502:], VaR_fore_3_99, ES_fore_3_99)
table_btES = pd.DataFrame()
table_btES['VaR and ES Forecast Methods'] = ['HS','FHS with EWMA', 'FHS with GARCH']*2
table_btES['Test Significance Level'] = ['95%']*3 + ['99%']*3
table_btES['Z'] = [btES_1_95, btES_2_95, btES_3_95, btES_1_99, btES_2_99, btES_3_99]
table_btES['p'] = [pES_1_95, pES_2_95, pES_3_95, pES_1_99, pES_2_99, pES_3_99]
table_btES


