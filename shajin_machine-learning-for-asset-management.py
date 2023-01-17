!pip install pydotplus
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import Ridge, ElasticNet, Lasso

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.tree import DecisionTreeRegressor, export_graphviz

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import mean_squared_error as MSE

from sklearn.metrics import r2_score, mean_absolute_error as MAE

import pydotplus as pdot

from IPython.display import Image

import matplotlib.pyplot as plt

from scipy.stats import norm

import scipy.stats

import matplotlib as mlp

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## importing Dataset



data = pd.read_csv('/kaggle/input/Data_Oct2018_v2.csv', index_col=0, parse_dates=True)[['World Equities',

                                                                                       'US Treasuries',

                                                                                       'Bond Risk Premium',

                                                                                       'Inflation Protection',

                                                                                       'Currency Protection',

                                                                                       'Real Estate']]



train = data.iloc[:int(0.85 * data.shape[0]), :]

test = data.iloc[int(0.85 * data.shape[0]):, :]

train.head()
test.shape
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





def compound(r):

    """

    returns the result of compounding the set of returns in r

    """

    return np.expm1(np.log1p(r).sum())



                         

def annualize_rets(r, periods_per_year):

    """

    Annualizes a set of returns

    We should infer the periods per year

    but that is currently left as an exercise

    to the reader :-)

    """

    compounded_growth = (1+r).prod()

    n_periods = r.shape[0]

    return compounded_growth**(periods_per_year/n_periods)-1





def annualize_vol(r, periods_per_year):

    """

    Annualizes the vol of a set of returns

    We should infer the periods per year

    but that is currently left as an exercise

    to the reader :-)

    """

    return r.std()*(periods_per_year**0.5)





def sharpe_ratio(r, riskfree_rate, periods_per_year):

    """

    Computes the annualized sharpe ratio of a set of returns

    """

    # convert the annual riskfree rate to per period

    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1

    excess_ret = r - rf_per_period

    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)

    ann_vol = annualize_vol(r, periods_per_year)

    return ann_ex_ret/ann_vol





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





def summary_stats(r, riskfree_rate=0.03):

    """

    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r

    """

    ann_r = r.aggregate(annualize_rets, periods_per_year=12)

    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)

    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)

    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())

    skew = r.aggregate(skewness)

    kurt = r.aggregate(kurtosis)

    cf_var5 = r.aggregate(var_gaussian, modified=True)

    hist_cvar5 = r.aggregate(cvar_historic)

    return pd.DataFrame({

        "Annualized Return": ann_r,

        "Annualized Vol": ann_vol,

        "Skewness": skew,

        "Kurtosis": kurt,

        "Cornish-Fisher VaR (5%)": cf_var5,

        "Historic CVaR (5%)": hist_cvar5,

        "Sharpe Ratio": ann_sr,

        "Max Drawdown": dd

    })
summary_stats(data)
data.aggregate(lambda r: drawdown(r).Drawdown.idxmin())
plt.style.use('seaborn-whitegrid')

(1 + data).cumprod().plot(figsize=(9, 8))

plt.show()
class linear_factor:

    """

linear_factor is used to create object for linear modelling of factors. Three types of models are accepted:

1. Lasso

2. Ridge

3. ElasticNet

    """

    def __init__(self, param, model_name):

        self.model_name = model_name

        self.param = param

    

    def get_model(self):

        if self.model_name == 'Lasso':

            return Lasso()

        elif self.model_name == 'ElasticNet':

            return ElasticNet()

        else:

            return Ridge()

    

    def param_tune(self, X, Y):

        """

        Tunes the hyperparameter of the given linear model, and saves the tuned model as object attribute. 

        Also, it returns the results of hyperparameter tuning.

        """

        model = self.get_model()

        grid_model = RandomizedSearchCV(model, n_iter = 10, param_distributions=self.param, scoring=['r2', 'neg_mean_squared_error'],

                                        cv=6, random_state=42, return_train_score=True, refit='r2', verbose = False).fit(X.to_numpy(), Y.to_numpy())

        self.columns = X.columns

        self.fit_model = grid_model.best_estimator_

        return pd.DataFrame(grid_model.cv_results_)[['param_alpha', 'mean_test_r2', 'mean_train_r2',

                                     'mean_test_neg_mean_squared_error', 'mean_train_neg_mean_squared_error']].sort_values(by = ['mean_test_r2'], axis = 0, ascending = False)

    

    def predict(self, X_test):

        self.y_pred = self.fit_model.predict(X_test.to_numpy())

        

    def summary(self, y_test):

        """

        Summarizes the results of the model and it's predictions.

        """

        y_test_n = y_test.to_numpy()

        mae = MAE(y_test_n, self.y_pred)

        mse = MSE(y_test_n, self.y_pred)

        r2 = r2_score(y_test_n, self.y_pred)

        print(self.model_name + '\'s Test Results:\nMSE = ' + str(mse) + ', MAE = ' + str(mae) + ', R-Square = ' + str(r2) + ',')

        print('Value to Fund manager:', round(self.fit_model.intercept_ * 100, 3), '%')

        

        coef = pd.DataFrame({'Factor Loadings' : self.fit_model.coef_}, index = self.columns)

        plt.style.use('Solarize_Light2')

        ax = coef.plot(kind = 'bar', figsize=(12, 10), color='green', legend = False)

        ax.set_title('Betas of Factors', y = 1.04)

        

        # set individual bar lables using above list

        for i in ax.patches:

            # get_x pulls left or right; get_height pushes up or down

            ax.text(i.get_x()+0.05, i.get_height()+.0025, \

                str(round(i.get_height()*100, 2))+'%', fontsize=15,

                    color='red')
param = {'alpha' : [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.00009]}
lasso = linear_factor(param, model_name = 'Lasso')

results = lasso.param_tune(train.drop('Real Estate', axis = 1), train['Real Estate'])

print('Validation Results:\nTrain MSE =', results['mean_train_neg_mean_squared_error'].mean(),

     ', Test MSE =', results['mean_test_neg_mean_squared_error'].mean())

results
lasso.predict(test.drop('Real Estate', axis = 1))

lasso.summary(test['Real Estate'])
ridge = linear_factor(param, model_name='Ridge')

results = ridge.param_tune(train.drop('Real Estate', axis = 1), train['Real Estate'])

print('Validation Results:\nTrain MSE =', results['mean_train_neg_mean_squared_error'].mean(),

     ', Test MSE =', results['mean_test_neg_mean_squared_error'].mean())  

results
ridge.predict(test.drop('Real Estate', axis = 1))

ridge.summary(test['Real Estate'])
param2 = {'alpha' : param['alpha'],

         'l1_ratio' : np.arange(0.01, 0.99, 0.01)}

elastic = linear_factor(param2, model_name='ElasticNet')

results = elastic.param_tune(train.drop('Real Estate', axis = 1), train['Real Estate'])

print('Validation Results:\nTrain MSE =', results['mean_train_neg_mean_squared_error'].mean(),

     ', Test MSE =', results['mean_test_neg_mean_squared_error'].mean())  

results
elastic.predict(test.drop('Real Estate', axis = 1))

elastic.summary(test['Real Estate'])