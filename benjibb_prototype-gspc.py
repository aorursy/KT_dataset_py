import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import os
print(os.listdir("../input"))
from scipy import stats
from scipy.stats import norm, skew #for some statistics
%matplotlib inline
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model
import warnings
warnings.warn = lambda *a, **kw: False
data = pd.read_csv("../input/GSPC.csv")
pd.to_datetime(data['Date'])
data = data.set_index('Date')
data.head()
data = data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
data['lrets'] = np.log(data['Adj Close']/data['Adj Close'].shift(1))
data.dropna(axis=0, inplace=True)
data.head()
def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 
# Before using log return
drawing1 = data['lrets'][-540:]
tsplot(np.diff(drawing1), lags=30)
# Financial returns are often heavy tailed, and a Student's T distribution is a simple method to capture this feature.
am = arch_model(data.lrets.values*100, p=3, o=0, q=3, dist='StudentsT', vol='EGARCH')
res = am.fit(update_freq=5, disp='off')
print(res.summary())
fig = res.plot(annualize='D')
_ = tsplot(res.resid, lags=30)
def _get_best_model(TS):
    best_aic = np.inf 
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl
# Convergence warnings can occur when dealing with very small numbers. 
# Multiplying the numbers by factors of 10 to scale the magnitude can help when necessary
TS = data['lrets'].values*100
best_aic, best_order, best_mdl = _get_best_model(TS)
# Using student T distribution usually provides better fit
# The optional inputs iter controls the frequency of output form the optimizer, 
# and disp controls whether convergence information is returned.
am = arch_model(TS, p=4, o=0, q=4, dist='StudentsT', vol='EGARCH')
res = am.fit(update_freq=5, disp='off')
print(res.summary())
drawing3 = res.resid[-540:]
tsplot(drawing3, lags=30)
# Daily
fig = res.plot(annualize='D')
# Monthly
fig = res.plot(annualize='M')
data.tail(20)
forecasts = res.forecast(horizon=1)
forecasts.residual_variance.iloc[-1]
