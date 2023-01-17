# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '../input/data-tkl'

df=pd.read_csv(f'{path}/Q15.csv')

df.tail()

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib.pylab as plt

plt.style.use('seaborn-dark-palette')
from sklearn.linear_model import LinearRegression

from plotnine import *

import matplotlib.pylab as pylab

from statsmodels.regression.linear_model import OLS
X=np.array(df[['K','L']].values)

y=np.array(df.Y)

import statsmodels.api as sma

x=sma.add_constant(X)

model=sma.OLS(y,x)

results=model.fit()

print('\nOLS线性模型回归R_square: \n',results.rsquared)

print('\nOLS线性模型回归R_adj: \n',results.rsquared_adj)

#print('\nOLS线性回归结果：\n',results.summary())
#fig, ax = plt.subplots(5,2, figsize=(12, 16), facecolor='#f7f7f7')

resid=pd.DataFrame(results.resid,columns=['resid'])

df_resid=pd.concat([df,resid],axis=1)

df_resid.tail()

df_plot=df_resid[['resid','K','L']]

#fig,ax=plt.subplots(1,2,figsize=(10,6))



(

ggplot(df_plot,aes(x='L',y='resid'))#

+ geom_point(alpha = 0.5, size = 2)

+theme(legend_position='none')+stat_smooth(color='blue')

#+ggtitle('Linear Regression of Tax versus GDP')

)



def reset_ramsey(res, degree=5):

    '''Ramsey's RESET specification test for linear models



    This is a general specification test, for additional non-linear effects

    in a model.





    Notes

    -----

    The test fits an auxiliary OLS regression where the design matrix, exog,

    is augmented by powers 2 to degree of the fitted values. Then it performs

    an F-test whether these additional terms are significant.



    If the p-value of the f-test is below a threshold, e.g. 0.1, then this

    indicates that there might be additional non-linear effects in the model

    and that the linear model is mis-specified.





    References

    ----------

    http://en.wikipedia.org/wiki/Ramsey_RESET_test



    '''

    order = degree + 1

    k_vars = res.model.exog.shape[1]

    #vander without constant and x:

    y_fitted_vander = np.vander(res.fittedvalues, order)[:, :-2] #drop constant

    exog = np.column_stack((res.model.exog, y_fitted_vander))

    res_aux = OLS(res.model.endog, exog).fit()

    #r_matrix = np.eye(degree, exog.shape[1], k_vars)

    r_matrix = np.eye(degree-1, exog.shape[1], k_vars)

    #df1 = degree - 1

    #df2 = exog.shape[0] - degree - res.df_model  (without constant)

    return res_aux.f_test(r_matrix) #, r_matrix, res_aux
reset_ramsey(results, degree=5)
df_new=np.log(df[['Y','K','L']])

df_new.tail()

X=np.array(df_new[['K','L']].values)

y=np.array(df_new.Y)

import statsmodels.api as sma

x=sma.add_constant(X)

model=sma.OLS(y,x)

results=model.fit()

#print('\nOLS线性回归结果：\n',results.summary())

print('\nOLS线性模型回归R_square: \n',results.rsquared)

print('\nOLS线性模型回归R_adj: \n',results.rsquared_adj)
#fig, ax = plt.subplots(5,2, figsize=(12, 16), facecolor='#f7f7f7')

resid=pd.DataFrame(results.resid,columns=['resid'])

df_resid=pd.concat([df,resid],axis=1)

df_resid.tail()

df_plot=df_resid[['resid','K','L']]

#fig,ax=plt.subplots(1,2,figsize=(10,6))



sns.set_style("darkgrid")

#sns.jointplot(x=df_plot["resid"],y=df_plot["K"], kind='scatter')

#sns.jointplot(x=df_plot["resid"],y=df_plot["L"], kind='scatter')



(

ggplot(df_plot,aes(x='L',y='resid'))#

+ geom_point(alpha = 0.5, size = 2)

+theme(legend_position='none')+stat_smooth(color='blue')

#+ggtitle('Linear Regression of Tax versus GDP')

)
reset_ramsey(results, degree=5)
path = '../input/question2'

df=pd.read_csv(f'{path}/Q2.csv')
df_new=df.copy()

df_new['Y']=np.log(df_new['Y'])

df_new['X']=np.log(df_new['X'])

df_new.rename(columns={"序号":"年份"},inplace=True)

df_new.tail()
X=np.array(df_new['X'].values)

y=np.array(df_new.Y)

import statsmodels.api as sma

x=sma.add_constant(X)

model=sma.OLS(y,x)

results=model.fit()

print('\nOLS线性模型回归R_square: \n',results.rsquared)

print('\nOLS线性模型回归R_adj: \n',results.rsquared_adj)

print('\nOLS线性回归结果：\n',results.summary())
import statsmodels

statsmodels.stats.diagnostic.acorr_breusch_godfrey(results,nlags=5)
df_new2=df_new.diff()

df_new2.drop(index=0,axis=0,inplace=True)

df_new2.head()

X=np.array(df_new2['X'].values)

y=np.array(df_new2.Y)

import statsmodels.api as sma

x=sma.add_constant(X)

model=sma.OLS(y,x)

results_robust=model.fit()

#print('\nOLS线性模型回归R_square: \n',results.rsquared)

#print('\nOLS线性模型回归R_adj: \n',results.rsquared_adj)

#print('\nOLS线性回归结果：\n',results.summary())

statsmodels.stats.diagnostic.acorr_breusch_godfrey(results_robust,nlags=5)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

print(plot_pacf(df_new['Y'],lags=15))
new = results.get_robustcov_results(cov_type='HAC',maxlags=1,use_correction=True, use_t=True)

print (new.summary())
from statsmodels.regression.linear_model import GLS

X=np.array(df_new[['X']].values)

y=np.array(df_new.Y)

x=sma.add_constant(X)

ols_resid = sma.OLS(y, x).fit().resid

res_fit = sma.OLS(ols_resid[1:], ols_resid[:-1]).fit()

rho = res_fit.params

from scipy.linalg import toeplitz

order = toeplitz(np.arange(34))

sigma = rho**order
model=sma.GLS(y,x,sigma=sigma)

results=model.fit()

print('\nGLS线性模型回归R_square: \n',results.rsquared)

print('\nGLS线性模型回归R_adj: \n',results.rsquared_adj)

print('\nGLS线性回归结果：\n',results.summary())