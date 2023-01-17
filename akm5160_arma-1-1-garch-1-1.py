import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns



data1=pd.read_csv('../input/AAPL_2006-01-01_to_2018-01-01.csv', parse_dates=['Date'], index_col='Date')

data1.head(5)
%matplotlib inline

plt.subplots(figsize=(14,4))

plt.plot(data1.Close)

data1.Volume.plot(secondary_y=True)
data1['ret']=0

data1['ret']=(data1.Close-data1.Close.shift(1))/data1.Close.shift(1)

plt.plot(data1.ret)
from statsmodels.tsa.stattools import adfuller

data1.ret.dropna(inplace=True)

adfuller(data1.ret)

#return series is stationary ,no -unit present

from statsmodels.tsa.stattools import acf, pacf





plt.subplots(figsize=(16,4))

plt.plot(data1.ret)
def stationarity(df):

    import statsmodels.api as sm

    acf_p=acf(data1.ret, nlags=10)

    pacf_p=pacf(data1.ret, nlags=10)

    f, ax = plt.subplots(2, 2, sharey=True, figsize=(17,6))

    ax[0,0].plot(df)

    ax[0,0].set_title('Sharing Y axis')

    

    ax[0,1].plot(acf_p)

    ax[0,1].axhline(y=0, linestyle='--', color='gray')

    ax[0,1].axhline(y=1.96/(np.sqrt(len(df))), linestyle='--', color='gray')

    ax[0,1].axhline(y=-1.96/(np.sqrt(len(df))), linestyle='--', color='gray')            

    ax[0,1].axhline(y=0, linestyle='--', color='gray')

    ax[0,1].set_title(str('ACF plot'))

    

    ax[1,0].plot(pacf_p)

    ax[1,0].axhline(y=0, linestyle='--', color='gray')

    ax[1,0].axhline(y=1.96/(np.sqrt(len(df))), linestyle='--', color='gray')

    ax[1,0].axhline(y=-1.96/(np.sqrt(len(df))), linestyle='--', color='gray')            

    ax[1,0].axhline(y=0, linestyle='--', color='gray')

    ax[1,0].set_title(str('PACF plot'))

    



    sm.qqplot(data1.ret, line='s', ax=ax[1,1])

    ax[1,1].set_title(str('QQ plot'))

    

    rs=adfuller(df)

    print (rs)

    

    plt.show()

    

    
stationarity(data1.ret)



data2=data1[2015:]

stationarity(data2.ret)
from statsmodels.tsa.stattools import ARMA

def best_AR_MA_checker(df,lower,upper):

    from statsmodels.tsa.stattools import ARMA

    from statsmodels.tsa.stattools import adfuller

    arg=np.arange(lower,upper)

    arg1=np.arange(lower,upper)

    best_param_i=0

    best_param_j=0

    temp=12000000

    rs=99

    for i in arg:

        for j in arg1:

            model=ARMA(df, order=(i,0,j))

            result=model.fit(disp=0)

            resid=adfuller(result.resid)

            if (result.aic<temp and  adfuller(result.resid)[1]<0.05):

                temp=result.aic

                best_param_i=i

                best_param_j=j

                rs=resid[1]

                

                

            print ("AR: %d, MA: %d, AIC: %d; resid stationarity check: %d"%(i,j,result.aic,resid[1]))

            

    print("the following function prints AIC criteria and finds the paramters for minimum AIC criteria")        

    print("best AR: %d, best MA: %d, best AIC: %d;  resid stationarity check:%d"%(best_param_i, best_param_j, temp, rs))     

best_AR_MA_checker(data2.ret,1,5)    
from  matplotlib import pylab



final_model=ARMA(data2.ret, order=(1,0,1))

result=final_model.fit(disp=0)

resid=adfuller(result.resid)

from statsmodels.stats.diagnostic import acorr_ljungbox

lbvalue, lbpvalue, bpvalue, bppvalue=acorr_ljungbox(np.square(result.resid), boxpierce=True)

#lbvalue, lbpvalue, bpvalue, bppvalue = diagnostic.acorr_ljungbox(np.square(residuals), boxpierce=True)



pylab.plot(lbpvalue, label="Ljung-Box") # Ljung-Box test  p-values (better for small smaples)

pylab.plot(bppvalue, label="Box-Pierce") # Box-Pierce test p-values (better for larger samples)

pylab.axhline(0.05, color="r")

pylab.legend()



data2.head(5)

from statsmodels.tsa.stattools import ARMA

model_f=ARMA(data2.ret, order=(1,1))

rs_f=model_f.fit(disp=0)

print (rs_f.summary())

data2['ma']=rs_f.resid

data2['ma_lag1']=data2['ma'].shift(1)



#ljung-box test for autocorrelation

from statsmodels.stats.diagnostic import acorr_ljungbox

#ar1,ar2,ar3,ar4=acorr_ljungbox(data2.ma**2, lags=10, boxpierce=True)

ar1,ar2,ar3,ar4=acorr_ljungbox(rs_f.resid**2, lags=10, boxpierce=True)

#2nd array list is p values from ljung-box test

#4th array list is p values from box-pierce test



plt.subplots(figsize=(12,4))

plt.plot(ar2, color='red')

plt.axhline(y=0)

plt.axhline(y=0.05)



#our intention is to check for autocorrelation, the part our mean model in ARMA was not able to capture.

#H0- No autocorrelation in the resids, no arch effect

#H1-autocorrelation in the residuals is present, Arch effect, model needs arch correction

#as we can see lag-1 and later lags are significant, so arch effect is present. p<alpha, reject the null hypothesis



#now we will be moving to arch correction via garch and we will try to model volatility.

data2['lag1']=data2['ret'].shift(1)

data2['lag2']=data2['ret'].shift(2)

data2.dropna(inplace=True)

#next step is building GARCH model and predict volatility and returns

#df=pd.DataFrame(pd.concat([data2.ma, data2.ma_lag1,data2.lag1], axis=1))

#import statsmodels.api as sm

#md=sm.OLS(data2.ret,df)

#md=md.fit()

#print( md.summary())



data2['rt-mue']=data2['ret']-data2.ret.mean()
data2.head(5)

ita=0.001

alpha=0.049

beta=0.95

plt.plot(data2['rt-mue'])

data2['sigma_sqr']=np.sqrt(data2.lag1**2)

data2['log_lik']=0



data2['sigma_sqr']=np.sqrt(ita+beta* (data2['sigma_sqr'].shift(1))**2+alpha*(data2.lag1)**2)

data2['log_lik']=-np.log(np.sqrt(2*3.14))-(data2.ret**2/(2*data2.sigma_sqr**2))-np.log(data2.sigma_sqr)

    

plt.subplots(figsize=(12,4))

plt.plot(data2.sigma_sqr)



#need to write a solver fucntion to maximize likelihiood sum while changing alpha, beta and theta 

#with the contraint alpha+beta+theta=1









plt.subplots(figsize=(12,4))

sns.distplot(md.resid)

#found using solver optimzation.

#need to deploy simpy for pythonic solver

ita1=0.021354

alpha1=0.978646

beta1=0.00

data2['sigma_sqr_f']=np.sqrt(data2.lag1**2)

data2['sigma_sqr_f']=np.sqrt(ita1+beta1* (data2['sigma_sqr_f'].shift(1))**2+alpha1*(data2.lag1)**2)
#found using solver optimzation.
