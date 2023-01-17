# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
stocks_df = pd.read_csv("../input/sandp500/all_stocks_5yr.csv", index_col = 0, parse_dates= True)
stocks_df.head()
stocks_df.tail()
unique_stocks_names = stocks_df.Name.unique()

unique_stocks_names
def select_and_plot_stock(stock):

  %matplotlib inline

  stock_df = stocks_df.loc[stocks_df.Name == stock, :]

  plt.figure(figsize = (10,10))

  plt.plot(stock_df.index, stock_df.close)

  plt.title(f"Stock Name : {stock}")

  plt.xlabel("Date")

  plt.ylabel("Close Price")

  plt.grid = True

  plt.show()

select_and_plot_stock("AAL")

select_and_plot_stock("AMZN")
stocks = unique_stocks_names

daily_return = pd.DataFrame() #create a DataFrame of daily returns

for n in range(len(stocks)):

        stock_df = stocks_df.loc[stocks_df.Name == stocks[n], :]

        stock_daily_return = stock_df["close"].pct_change() #usring pct_change function can give us the daily return

        daily_return[stocks[n]] = stock_daily_return

        n += 1

daily_return = daily_return.drop(daily_return.index[0],axis=0) #drop the first column which doesn't have data in daily return data frame

daily_return
def plot_daily_return(stock_names):

    for n in range(len(stock_names)):

        plt.figure(figsize = (20,10))

        plt.subplot(len(stock_names),1,n+1)

        plt.plot(daily_return.index, daily_return[stock_names[n]])

        plt.title(f"Stock Name : {stock_names[n]}")

        plt.xlabel("Date")

        plt.ylabel("Daily Return")

        plt.grid = True

        plt.show()

        plt.grid = True

        n+=1
names = ["AMZN", "EBAY", "AAL"]

plot_daily_return(names)
def dist_plot_return(stock_names):

    for n in range(len(stock_names)):

        plt.figure(figsize = (10,10))

        plt.subplot(len(stock_names),1,n+1)

        daily_return_stock = daily_return.loc[:, stock_names[n]]

        daily_return_stock.plot.hist(bins = 50)

        plt.title(f"Stock Name : {stock_names[n]}")

        plt.xlabel("Date")

        plt.ylabel("Daily Return")

        plt.grid = True

        plt.show()

        plt.grid = True

        n+=1
names = ["AMZN", "EBAY", "AAL"]

dist_plot_return(names)
def annualize_rets(r, periods_per_year):

  """

  Annulalizes a set of returns

  """



  compounded_grouwth = (1+r).prod()

  n_periods = r.shape[0]

  return compounded_grouwth**(periods_per_year/n_periods)-1


annualize_rets(daily_return["AMZN"], 252) #The operational days of the stocks market in a year is 252 days
def drawdown(returns_series: pd.Series):

  """Takes a time series of asset returns. 

     returns a DataFrame with columns for the wealth index,

     the previous peaks, and

     the percentage drawdown

  """

  wealth_index = 1000*(1+returns_series).cumprod()

  previous_peaks = wealth_index.cummax()

  drawdowns = (wealth_index - previous_peaks)/previous_peaks

  return pd.DataFrame({"Wealth" : wealth_index, "Previous Peak" : previous_peaks, "Drawdown" : drawdowns})



drawdown(daily_return["AMZN"])
plt.figure(figsize = (10, 10))

drawdown(daily_return["AMZN"])[["Wealth", "Previous Peak"]].plot()

plt.show()
plt.figure(figsize = (10, 5))

drawdown(daily_return["AMZN"])["Drawdown"].plot()
def skewness (r):

  """

  Alternative to scipy.stats.skew()

  computes the skewness of the supplied series or dataframe

  returns a float or series

  """

  demeaned_r = r - r.mean()

  #use the population standard deviation, so set dof=0

  sigma_r = r.std(ddof=0)

  exp = (demeaned_r ** 3).mean()

  return exp/sigma_r ** 3
skewness_df = pd.DataFrame(skewness(daily_return))

skewness_df.columns = ["Skewness"]

skewness_df
for n in range (0,506, 30):

    sample_skewness_df = skewness_df[n : n+20].sort_values(by = ["Skewness"])

    %matplotlib inline

    sample_skewness_df.plot.bar()

    plt.show()
def kurtosis(r):

  demeaned_r = r - r.mean()

  sigma_r = r.std(ddof=0)

  exp = (demeaned_r**4).mean()

  return exp/sigma_r**4

kurtosis_df = pd.DataFrame(kurtosis(daily_return))

kurtosis_df.columns = ["Kurtosis"]

kurtosis_df
for n in range (0,506, 30):

    sample_kurtosis_df = kurtosis_df[n : n+20].sort_values(by = ["Kurtosis"])

    %matplotlib inline

    sample_kurtosis_df.plot.bar()

    plt.show()
er = annualize_rets(daily_return[:"2015"],252) #annualized expected return in the 2013 to 2015 period

er
cov = daily_return.cov()

cov.head()
def portfolio_return(weights, returns):

  """

  weights --> Returns

  """

  return weights.T @ returns # transposing the weights matrix and multiply it by returns

def portfolio_vol(weights, covmat):

  """

  Weights --> Vol

  """

  return (weights.T @ covmat @ weights)**0.5
from scipy.optimize import minimize

def minimize_vol(target_return, er, cov):

  """

  target_ret ==> W 

  """

  n = er.shape[0] #number of weights

  init_guess = np.repeat(1/n, n) #makes a tuple of n tuples for weights

  bounds = ((0.0, 1.0),)*n #defines the max and min of our possible weights

  return_is_target = {

      "type" : "eq",

      "args" : (er,),

      "fun" : lambda weights, er :

       target_return - portfolio_return(weights, er)

  } 

    

  weights_sum_to_one = {

      "type" : "eq",

      "fun" : lambda weights: np.sum(weights) - 1

      }

  results = minimize(

      portfolio_vol, init_guess,

                     args = (cov,), method = "SLSQP",

                     options = {"disp" : False},

                     constraints = (return_is_target, weights_sum_to_one),

                     bounds = bounds

                    )

  return results.x
def optimal_weights(n_points, er, cov):

  """

  Generates a list of weights to run the optimizer on, to minimize the volatility

  """

  target_rs = np.linspace(er.min(), er.max(), n_points)

  weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]

  return weights
def msr(riskfree_rate, er, cov):

  """

  Returns the weights of the portfolio that gives you the maximum sharpe ratio given the riskfree rate and expected returns and a covariancce matrix

  """

  n = er.shape[0] 

  init_guess = np.repeat(1/n, n) 

  bounds = ((0.0, 1.0),)*n 

  weights_sum_to_one = {

      "type" : "eq",

      "fun" : lambda weights: np.sum(weights) - 1

  }

  def neg_sharpe_ratio(weights, riskfree_rate, er, cov):

    """

    Returns the negative of the sharpe ratio, given weights

    """

    r = portfolio_return(weights, er) 

    vol = portfolio_vol(weights, cov) 

    return -(r - riskfree_rate)/vol



  results = minimize(neg_sharpe_ratio, init_guess,

                     args=(riskfree_rate, er, cov,), method="SLSQP",

                     options={"disp" : False},

                     constraints = (weights_sum_to_one),

                     bounds=bounds

                    )

  return results.x
def gmv(cov):

  """

  Returns the weights of the Global Minimum Vol Portfolio given covariance matrix

  """

  n = cov.shape[0]

  return msr(0, np.repeat(1,n), cov)
def plot_ef(n_points, er, cov, show_cml=False, show_ew=False, show_gmv=False, riskfree_rate=0, style=".-"):

  """

  Plots the efficient frontier curve, msr, gmv, and ew points.

  """

  weights = optimal_weights(n_points, er, cov)

  rets = [portfolio_return(w, er) for w in weights]

  vols = [portfolio_vol(w, cov) for w in weights]

  ef = pd.DataFrame({"Returns" : rets, "Volatility" : vols})

  ax = ef.plot.line(x="Volatility", y = "Returns", style = style)

  if show_gmv:

    w_gmv = gmv(cov)

    r_gmv = portfolio_return(w_gmv, er)

    vol_gmv = portfolio_vol(w_gmv, cov)

    #displat EW

    ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker = "o", markersize=12)

  if show_ew:

    n = er.shape[0]

    w_ew = np.repeat(1/n , n)

    r_ew = portfolio_return(w_ew, er)

    vol_ew = portfolio_vol(w_ew, cov)

    #displat EW

    ax.plot([vol_ew], [r_ew], color="red", marker = "o", markersize=10)  

  if show_cml:

    ax.set_xlim(left = 0)

    rf = 0.1

    w_msr = msr(riskfree_rate, er=er, cov=cov)

    r_msr = portfolio_return(w_msr, er)

    vol_msr = portfolio_vol(w_msr, cov)

    #Add CML

    cml_x = [0, vol_msr]

    cml_y = [riskfree_rate, r_msr]

    ax.plot(cml_x, cml_y, color = "green", marker = "o", linestyle = "dashed", markersize = 12, linewidth =2)

  return ax

plot_ef(20, er, cov, show_cml=True, show_ew=True, show_gmv=True, riskfree_rate=0.03)
ew_weights = np.repeat(1/er.shape[0], er.shape[0])

ew_weights[:30] # We can see the firts 30 allocated weights

portfolio_return(ew_weights, er)
msr_weights = msr(0.03, er, cov)

msr_weights[:30]
portfolio_return(msr_weights, er)
gmv_weights = gmv(cov)

gmv_weights[:30]
portfolio_return(gmv_weights, er)
rets = annualize_rets(daily_return["2016":],252)

rets
portfolio_return(ew_weights, rets)
portfolio_return(msr_weights, rets)
portfolio_return(gmv_weights, rets)