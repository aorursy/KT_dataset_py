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
from scipy.optimize import minimize

import matplotlib.pyplot as plt



def drawdown(return_series: pd.Series):

    """Takes a time series of asset returns.

       returns a DataFrame with columns for

       the wealth index, 

       the previous peaks, and 

       the percentage drawdown

    """

    wealth_index = 1000*(1+return_series).cumprod()

    previous_peaks = wealth_index.cummax()

    drawdowns = (wealth_index - previous_peaks)/previous_peaks

    return pd.DataFrame({"Wealth": wealth_index, 

                         "Previous Peak": previous_peaks, 

                         "Drawdown": drawdowns})

   

def skewness(r):

    """

    Alternative to scipy.stats.skew()

    Computes the skewness of the supplied Series or DataFrame

    Returns a float or a Series

    """

    demeaned_r = r - r.mean()

    # use the population standard deviation, so set dof=0

    sigma_r = r.std(ddof=0)

    exp = (demeaned_r**3).mean()

    return exp/sigma_r**3





def kurtosis(r):

    """

    Alternative to scipy.stats.kurtosis()

    Computes the kurtosis of the supplied Series or DataFrame

    Returns a float or a Series

    """

    demeaned_r = r - r.mean()

    # use the population standard deviation, so set dof=0

    sigma_r = r.std(ddof=0)

    exp = (demeaned_r**4).mean()

    return exp/sigma_r**4



import scipy.stats

def is_normal(r, level=0.01):

    """

    Applies the Jarque-Bera test to determine if a Series is normal or not

    Test is applied at the 1% level by default

    Returns True if the hypothesis of normality is accepted, False otherwise

    """

    if isinstance(r, pd.DataFrame):

        return r.aggregate(is_normal)

    else:

        statistic, p_value = scipy.stats.jarque_bera(r)

        return p_value > level

    

    

    

    

def semideviation(r):

    """

    Returns the semideviation aka negative semideviation of r

    r must be a Series or a DataFrame

    """

    is_negative = r < 0

    return r[is_negative].std(ddof=0)





def var_historic(r, level=5):

    """

    Returns the historic Value at Risk at a specified level

    i.e. returns the number such that "level" percent of the returns

    fall below that number, and the (100-level) percent are above

    """

    if isinstance(r, pd.DataFrame):

        return r.aggregate(var_historic, level=level)

    

    elif isinstance(r, pd.Series):

        return -np.percentile(r, level)

    else:

        raise TypeError("Expected r to be a Series or DataFrame")





def cvar_historic(r, level=5):

    """

    Computes the Conditional VaR of Series or DataFrame

    """

    if isinstance(r, pd.Series):

        is_beyond = r <= -var_historic(r, level=level)

        return -r[is_beyond].mean()

    elif isinstance(r, pd.DataFrame):

        return r.aggregate(cvar_historic, level=level)

    else:

        raise TypeError("Expected r to be a Series or DataFrame")





from scipy.stats import norm

def var_gaussian(r, level=5, modified=False):

    """

    Returns the Parametric Gauusian VaR of a Series or DataFrame

    If "modified" is True, then the modified VaR is returned,

    using the Cornish-Fisher modification

    """

    # compute the Z score assuming it was Gaussian

    z = norm.ppf(level/100)

    if modified:

        # modify the Z score based on observed skewness and kurtosis

        s = skewness(r)

        k = kurtosis(r)

        z = (z +

                (z**2 - 1)*s/6 +

                (z**3 -3*z)*(k-3)/24 -

                (2*z**3 - 5*z)*(s**2)/36

            )

    return -(r.mean() + z*r.std(ddof=0))





def annualize_rets(r, periods_per_year):

    """

    Annualizes a set of returns

    

    """

    compounded_growth = (1+r).prod()

    n_periods = r.shape[0]

    return compounded_growth**(periods_per_year/n_periods)-1





def annualize_vol_using_ret(returns, period):

    a = returns.std()*(period**0.5)

    return a



def annualize_vol_compiled(vol, period):

    a = vol*(period**0.5)

    return a





def semideviation(r):

    """

    Returns the semideviation aka negative semideviation of r

    r must be a Series or a DataFrame, else raises a TypeError

    """

    if isinstance(r, pd.Series):

        is_negative = r < 0

        return r[is_negative].std(ddof=0)

    elif isinstance(r, pd.DataFrame):

        return r.aggregate(semideviation)

    else:

        raise TypeError("Expected r to be a Series or DataFrame")

        

        

def get_ind_returns():

    """

    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns

    """

    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0)/100

    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')

    ind.columns = ind.columns.str.strip()

    return ind



def transform_to_executable(ind, formating = "%Y%m"):

    ind.index = pd.to_datetime(ind.index, format=formating).to_period('M')

    ind.columns = ind.columns.str.strip()

    return ind





def portfolio_return(weights, returns):

    """

    Computes the return on a portfolio from constituent returns and weights

    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix

    """

    return weights.T @ returns







def portfolio_vol(weights, covmat):

    """

    Computes the vol of a portfolio from a covariance matrix and constituent weights

    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix

    """

    return (weights.T @ covmat @ weights)**0.5





def plot_ef2(n_points, er, cov):

    """

    Plots the 2-asset efficient frontier

    """

    import numpy as np

    if er.shape[0] != 2 or er.shape[0] != 2:

        raise ValueError("plot_ef2 can only plot 2-asset frontiers")

    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]

    rets = [portfolio_return(w, er) for w in weights]

    vols = [portfolio_vol(w, cov) for w in weights]

    ef = pd.DataFrame({

        "Returns": rets, 

        "Volatility": vols

    })

    return ef.plot.line(x="Volatility", y="Returns", style=".-")







def minimize_vol(target_return, er, cov):

    

    """

    Returns the optimal weights that achieve the target return

    given a set of expected returns and a covariance matrix

    """

    import numpy as np

    from scipy.optimize import minimize

    n = er.shape[0]

    init_guess = np.repeat(1/n, n)

    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!

    # construct the constraints

    weights_sum_to_1 = {'type': 'eq',

                        'fun': lambda weights: np.sum(weights) - 1

    }

    return_is_target = {'type': 'eq',

                        'args': (er,),

                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)

    }

    weights = minimize(portfolio_vol, init_guess,

                       args=(cov,), method='SLSQP',

                       options={'disp': False},

                       constraints=(weights_sum_to_1,return_is_target),

                       bounds=bounds)

    return weights.x











def optimal_weights(n_points, er, cov):

    """

    n_points = number of portfolios generated

    er = annualized returns

    cov = coveriance matrix

    """

    target_rs = np.linspace(er.min(), er.max(), n_points)

    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]

    return weights





def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):

    """

    Plots the multi-asset efficient frontier

    """

    weights = optimal_weights(n_points, er, cov)

    rets = [portfolio_return(w, er) for w in weights]

    vols = [portfolio_vol(w, cov) for w in weights]

    ef = pd.DataFrame({

        "Returns": rets, 

        "Volatility": vols

    })

    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)

    if show_cml:

        ax.set_xlim(left = 0)

        # get MSR

        w_msr = msr(riskfree_rate, er, cov)

        r_msr = portfolio_return(w_msr, er)

        vol_msr = portfolio_vol(w_msr, cov)

        

        # add CML

        cml_x = [0, vol_msr]

        cml_y = [riskfree_rate, r_msr]

        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)

    if show_ew:

        n = er.shape[0]

        w_ew = np.repeat(1/n, n)

        r_ew = portfolio_return(w_ew, er)

        vol_ew = portfolio_vol(w_ew, cov)

        

        # add EW

        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)

    if show_gmv:

        w_gmv = gmv(cov)

        r_gmv = portfolio_return(w_gmv, er)

        vol_gmv = portfolio_vol(w_gmv, cov)

        

        # add EW

        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)

        

        return ax

    

    

    

def get_portfolios(er, cov, show_cml=True, riskfree_rate=0, show_ew=True, show_gmv=True):

     '''

     er -> annulized returns

     cov -> covariance matrix of the returns

    

     Gets the portfolio weights:

     Maximum shape ratio

     Global minimum variance

     Equally weighted

     '''

    

     if show_cml:

        # get MSR

        w_msr = msr(riskfree_rate, er, cov)

        r_msr = portfolio_return(w_msr, er)

        vol_msr = portfolio_vol(w_msr, cov)

        msrD = pd.DataFrame(data = np.round(w_msr,3), index = er.index)

        print("MSR portfolio is:")

        #print(np.round(w_msr,3))

        print(msrD)

        msrD.plot.bar(figsize=(15, 6), title = "MSR")

        

    

    

     if show_gmv:

        w_gmv = gmv(cov)

        r_gmv = portfolio_return(w_gmv, er)

        vol_gmv = portfolio_vol(w_gmv, cov)

        gmvD = pd.DataFrame(data = np.round(w_gmv,3), index = er.index)

        print("GMV protfolio is:")

        #print(np.round(w_gmv,4))

        print(gmvD)

        gmvD.plot.bar(figsize=(15, 6), title = "GMW")

        

     if show_ew:

        n = er.shape[0]

        w_ew = np.repeat(1/n, n)

        r_ew = portfolio_return(w_ew, er)

        vol_ew = portfolio_vol(w_ew, cov)

        ewD = pd.DataFrame(data = np.round(w_ew,3), index = er.index)

        print("EW portfolio is:")

        #print(np.round(w_ew,2))

        print(ewD)

        ewD.plot.bar(figsize=(15, 6), title = "EW")

        

     



        

    

    



def get_oreturns_volatility_table(n_points, er, cov):

    

    weights = optimal_weights(n_points, er, cov)

    rets = [portfolio_return(w, er) for w in weights]

    vols = [portfolio_vol(w, cov) for w in weights]

    ef = pd.DataFrame({

        "Returns": rets, 

        "Volatility": vols

    })

    return ef





def plot_risk_averse_portfolio(volatility_table):

    a = volatility_table

    plt.axvline(x=min(a['Volatility']), color = 'red')

    plt.plot(a['Volatility'],a['Returns'])

    plt.show()

    val = a[a['Volatility'] == min(a['Volatility'])]

    print ("Minimum volatility in " + str(round(min(a['Volatility']),3)))

    print ("The coresponding return is " + str(round(val.iloc[0]['Returns'],3)))



    

    

def msr(riskfree_rate, er, cov):

    """

    Returns the weights of the portfolio that gives you the maximum sharpe ratio

    given the riskfree rate and expected returns and a covariance matrix

    """

    n = er.shape[0]

    init_guess = np.repeat(1/n, n)

    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!

    # construct the constraints

    weights_sum_to_1 = {'type': 'eq',

                        'fun': lambda weights: np.sum(weights) - 1

    }

    def neg_sharpe(weights, riskfree_rate, er, cov):

        """

        Returns the negative of the sharpe ratio

        of the given portfolio

        """

        r = portfolio_return(weights, er)

        vol = portfolio_vol(weights, cov)

        return -(r - riskfree_rate)/vol



    weights = minimize(neg_sharpe, init_guess,

                       args=(riskfree_rate, er, cov), method='SLSQP',

                       options={'disp': False},

                       constraints=(weights_sum_to_1,),

                       bounds=bounds)

    return weights.x

    

def gmv(cov):

    """

    Returns the weights of the Global Minimum Volatility portfolio

    given a covariance matrix

    """

    n = cov.shape[0]

    return msr(0, np.repeat(1, n), cov)



def portfolio_destrib(target_return, retPerAn, covarianceMatrix):

    minvol = minimize_vol(target_return, retPerAn, covarianceMatrix)

    msrD = pd.DataFrame(data = np.round(minvol,3), index = retPerAn.index)

    print(msrD)

    msrD.plot.bar(figsize=(15, 6), title = "MSR")
!pip install yfinance
import yfinance as yf

import pandas as pd

import numpy as np





#define the ticker symbol

tickerSymbol = ['MSFT','AAPL', 'AMD', 'BRK-B']
#get data on this ticker

Stocks = pd.DataFrame()

Stocks.index.names = ['Date']
for ticker in tickerSymbol:

    tickerData = yf.download(ticker,'2015-01-01','2020-05-01')

    tickerData = pd.DataFrame([tickerData['Close']]).T

    tickerData.columns = [ticker]

    

    Stocks = pd.merge(Stocks,tickerData,on='Date',how='right')
Stocks
returns = Stocks/Stocks.shift(1) - 1

returns
lifetime_ret = (returns+1).prod()-1

lifetime_ret
retPerAn = annualize_rets(returns, 252)

retPerAn
covarianceMatrix  = returns.cov()

covarianceMatrix
plot_ef(20, retPerAn, covarianceMatrix, legend=True, show_gmv=True, show_ew=True)
get_portfolios(retPerAn, covarianceMatrix)
portfolio_destrib(0.25, retPerAn, covarianceMatrix)
portfolio_return(gmv(covarianceMatrix),retPerAn)