import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
os.listdir("../input")
raw_stocks = pd.HDFStore("../input/stocks.hdf")
keys = raw_stocks.keys()
stocks = pd.DataFrame()
stocks["Date"]  = raw_stocks['/AAIGF']["Close"].index
stocks.index = stocks["Date"]
for key in keys:

    stocks[key[1:]] = raw_stocks[key]['Close']
stocks.pop("Date");
stocks.shape
stocks.head()
stocks = stocks.dropna(axis=1,how='any')
stocks.shape
stocks.head()
lagged_stocks = stocks.shift()

lagged_stocks.head()
returns = (stocks - lagged_stocks) / (lagged_stocks)

returns = returns.dropna()
returns["market_mean"] = returns.mean(axis=1)

returns["market_std"] = returns.std(axis=1)
returns.head()
plt.figure(figsize=(50,50))

sns.heatmap(returns.corr())

plt.show()
fig = plt.figure(figsize=(15,8))



ax1 = fig.add_subplot(2,3,1) 

plt.hist(returns["market_mean"],bins=100,label=("Market"))

ax1.title.set_text('Market')



ax2 = fig.add_subplot(2,3,2) 

plt.hist(returns["AAPL"],bins=100)

ax2.title.set_text('Apple')



ax3 = fig.add_subplot(2,3,3) 

plt.hist(returns["AMZN"],bins=100)

ax3.title.set_text('Amazon')



ax4 = fig.add_subplot(2,3,4) 

plt.hist(returns["GOOGL"],bins=100)

ax4.title.set_text('Google')



ax5 = fig.add_subplot(2,3,5) 

plt.hist(returns["MSFT"],bins=100)

ax5.title.set_text('Microsoft')



ax5 = fig.add_subplot(2,3,6) 

plt.hist(returns["SAP"],bins=100)

ax5.title.set_text('SAP')



plt.show()
plt.figure(figsize=(20,5))

plt.plot(returns["market_mean"])

plt.title("returns du market")

plt.show()
from sklearn.decomposition import PCA
pca= PCA(n_components=100)
returns_pca = pca.fit_transform(returns)
plt.figure(figsize=(20,5))

plt.plot(returns.index,returns_pca[:,0].ravel())

plt.title("composante principale des returns sous PCA")

plt.show()
plt.plot(pca.explained_variance_ratio_)

plt.show()
plt.figure(figsize=(20,5))

returns['sharpe_ratio'] = returns["market_mean"] / (returns["market_std"] * np.sqrt(260))

plt.plot(returns["sharpe_ratio"])

plt.title("sharpe ratios du market")

plt.show()
autocorrs = []



for i in range(1,252):

    xt = returns['market_mean']

    xs = returns['market_mean'].shift(i).dropna()

    

    xt = xt - xt.mean()

    xs = xs - xs.mean()

       

    ct = (xt * xs).mean() / (xt.std()*xs.std())



    autocorrs.append(ct)



plt.figure(figsize=(20,5))



plt.plot(autocorrs)

plt.title("Autocorrélations du market")

plt.show()
from statsmodels.tsa.stattools import pacf

plt.figure(figsize=(20,5))

pcor = pacf(returns['market_mean'],nlags=252)

plt.plot(pcor[1:])

plt.title("Autocorrélations partielles du market")

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.filters import hp_filter





plt.plot(returns['market_mean'])

plt.title('market mean')

plt.show()



decomp = seasonal_decompose(returns['market_mean'], model='additive',freq=365)

fig = decomp.plot()

plt.title("Decomposition saisonnier classique")

plt.show()



cycle, trend = hp_filter.hpfilter(X=returns['market_mean'],lamb=100*(365**2))

#plt.plot(returns[colname],c='blue',alpha=0.3,label='actual')

plt.plot(trend,c='red',alpha=0.3,label='trend')

plt.plot(cycle,c='green',alpha=0.3,label='cycle')

plt.legend()

plt.title("Hodrick-Prescott")

plt.show()
import pywt

cA, cD = pywt.dwt(returns['market_mean'], 'db1')



plt.plot(cA,c='red',alpha=0.3,label='cA')

plt.plot(cD,c='green',alpha=0.3,label='cD')

plt.legend()

plt.title("Wavelet")

plt.show()
nlag = 365



x = returns["market_mean"]



for i in range(1,nlag):

    xs = returns['market_mean'].shift(i)

    x = pd.concat([x,xs],axis=1)

    

cols = ["market_mean_lag_" + str(i) for i in range(nlag)]

x.columns = cols

x = x.dropna(axis=0,how='any')

y = x.pop("market_mean_lag_0")
y.index  = x.index
xtrain = x[x.index < '2017-01-01']

xtest = x[x.index >= '2017-01-01']



ytrain = y[y.index < '2017-01-01']

ytest = y[y.index >= '2017-01-01']
from sklearn.linear_model import Lasso,LinearRegression

from sklearn.ensemble import RandomForestRegressor
lr = LinearRegression()

ls = Lasso(alpha=0.1)

rf = RandomForestRegressor()
lr.fit(xtrain,ytrain)

ls.fit(xtrain,ytrain)

rf.fit(xtrain,ytrain);
lr_preds = lr.predict(xtest)

ls_preds = ls.predict(xtest)

rf_preds = rf.predict(xtest);
from sklearn.metrics import r2_score,mean_squared_error

lr_r2 = r2_score(y_true=ytest, y_pred=lr_preds)

lr_mse = mean_squared_error(ytest,lr_preds)

print("R2: {}, MSE:{}".format(lr_r2,lr_mse))
ls_r2 = r2_score(y_true=ytest, y_pred=ls_preds)

ls_mse = mean_squared_error(ytest,ls_preds)

print("R2: {}, MSE:{}".format(ls_r2,ls_mse))
rf_r2 = r2_score(y_true=ytest, y_pred=rf_preds)

rf_mse = mean_squared_error(ytest,rf_preds)

print("R2: {}, MSE:{}".format(rf_r2,rf_mse))
preds = pd.concat([pd.DataFrame(lr_preds),pd.DataFrame(ls_preds)

                  ,pd.DataFrame(rf_preds)],axis=1)

preds.index = ytest.index

preds.columns = ['lr','ls','rf']
plt.figure(figsize=(20,5))

plt.plot(ytrain,c='blue')

plt.plot(ytest,c='red')

plt.plot(preds['lr'],c='green')

plt.title("Linear Regression")

plt.show()



plt.figure(figsize=(20,5))

plt.plot(ytrain,c='blue')

plt.plot(ytest,c='red')

plt.plot(preds['ls'],c='orange')

plt.title("Lasso Regression")

plt.show()



plt.figure(figsize=(20,5))

plt.plot(ytrain,c='blue')

plt.plot(ytest,c='red')

plt.plot(preds['rf'],c='purple')

plt.title("Random Forest")

plt.show()



ls.coef_
#Markowitz
def rand_weights(n):

    w = np.random.rand(n)

    return (w / sum(w))
def rand_portfolio(returns_matrix):

    p =  np.asmatrix(np.mean(returns_matrix,axis=0))

    w =  np.asmatrix(rand_weights(returns_matrix.shape[1]))

    C =  np.asmatrix(np.cov(returns_matrix.T))

    

    mu = w * p.T

    sigma = np.sqrt(w * C * w.T)

    

    return mu,sigma,w
def gen_portfolio(n,returns_matrix):

    means=[]

    stds=[]

    weights=[]

    sharps=[]

    

    for i in range(n):

        m,s,w = rand_portfolio(returns_matrix)

        means.append(m)

        stds.append(s)

        weights.append(w)

        sharps.append(m/s)

    

    return np.array(means),np.array(stds),np.array(weights),np.array(sharps)
n_portfolios = 1000

means, stds, weights,sharps = gen_portfolio(n_portfolios,returns)
sharp_max = np.argmax(sharps)

mu_max = means[sharp_max]

sigma_max = stds[sharp_max]

weight_max = weights[sharp_max]
# 1/n
p =  np.asmatrix(np.mean(returns,axis=0))

w =  np.asmatrix(np.ones(returns.shape[1]) / returns.shape[1])

C =  np.asmatrix(np.cov(returns.T))

    

mu = w * p.T

sigma = np.sqrt(w * C * w.T)
plt.scatter(stds,means,c='blue')

plt.xlim(0.0077,0.0093)

plt.ylim(0.0003,0.0008)

plt.scatter(np.array(sigma),np.array(mu),c='orange',label='1/n')

plt.scatter(np.array(sigma_max),np.array(mu_max),c='red',label='Markowitz')

plt.legend(loc='lower right')

plt.title("Risque/Rendement")

plt.show()
#Avec Predicteurs ARIMA
!pip install pmdarima



from pmdarima.arima import auto_arima
ret_arima_preds = pd.DataFrame()



for colname in returns.columns:

    ret = returns[colname]

    # fit stepwise auto-ARIMA

    ret_arima = auto_arima(ret,max_p=31,max_q=31,

                           seasonal = False,max_order = 62,

                           trace=False,

                           error_action='ignore',  # don't want to know if an order does not work

                           suppress_warnings=True,  # don't want convergence warnings

                           stepwise=True)  # set to stepwise

    

    ret_arima_preds = pd.concat([ret_arima_preds,pd.DataFrame(ret_arima.predict(n_periods = 7))],axis=1)



ret_arima_preds.shape
ret_arima_preds.columns = returns.columns



from datetime import timedelta

preds_dates = [returns.index[-1] + timedelta(days=i) for i in range(1,8)]

ret_arima_preds.index = preds_dates



ret_arima_preds.head()
stock = np.random.choice(ret_arima_preds.columns)

plt.plot(returns[stock][returns.index[-60]:returns.index[-1]])

plt.plot(ret_arima_preds[stock])

plt.title(stock)

plt.xticks(rotation=45)

plt.show()
#Résultat
n_portfolios = 1000

ameans, astds, aweights,asharps = gen_portfolio(n_portfolios,ret_arima_preds)



asharp_max = np.argmax(asharps)

amu_max = ameans[asharp_max]

asigma_max = astds[asharp_max]

aweight_max = aweights[asharp_max]
print("rendement prédit:" + str(amu_max[0][0]) )

print("risque prédit:" + str(asigma_max[0][0]) )
eq_pnl = (np.asmatrix(w) * np.asmatrix(returns.T)).cumsum()

eq_pnl = np.array(eq_pnl).ravel()
maxsharpe_pnl = (np.asmatrix(weight_max) * np.asmatrix(returns.T)).cumsum()

maxsharpe_pnl = np.array(maxsharpe_pnl).ravel()
plt.plot(returns.index,eq_pnl/eq_pnl.std(),label='1/n')

plt.plot(returns.index,maxsharpe_pnl/maxsharpe_pnl.std(),label='Markowitz')

plt.legend()

plt.title("PNL")

plt.show()
arima_pnl = (np.asmatrix(aweight_max) * np.asmatrix(ret_arima_preds.T)).cumsum()

arima_pnl = np.array(arima_pnl).ravel()
plt.plot(ret_arima_preds.index,arima_pnl/arima_pnl.std(),label='ML')

plt.legend()

plt.title("PNL")

plt.xticks(rotation=45)

plt.show()