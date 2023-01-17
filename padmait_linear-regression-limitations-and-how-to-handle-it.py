# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_boston



# load data

boston = load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)

X.drop('CHAS', axis=1, inplace=True)

y = pd.Series(boston.target, name='MEDV')



# inspect data

X.head()
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X, y)



print(f'Coefficients: {lin_reg.coef_}')

print(f'Intercept: {lin_reg.intercept_}')

print(f'R^2 score: {lin_reg.score(X, y)}')
import statsmodels.api as sm



X_constant = sm.add_constant(X)

lin_reg = sm.OLS(y,X_constant).fit()

lin_reg.summary()
%matplotlib inline

%config InlineBackend.figure_format ='retina'

import seaborn as sns 

import matplotlib.pyplot as plt

import statsmodels.stats.api as sms

sns.set_style('darkgrid')

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)



def linearity_test(model, y):

    '''

    Function for visually inspecting the assumption of linearity in a linear regression model.

    It plots observed vs. predicted values and residuals vs. predicted values.

    

    Args:

    * model - fitted OLS model from statsmodels

    * y - observed values

    '''

    fitted_vals = model.predict()

    resids = model.resid



    fig, ax = plt.subplots(1,2)

    

    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})

    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)

    ax[0].set(xlabel='Predicted', ylabel='Observed')



    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})

    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)

    ax[1].set(xlabel='Predicted', ylabel='Residuals')

    

linearity_test(lin_reg, y)    

    
# Check the mean of the residuals if its zero

lin_reg.resid.mean() # -1.0012544153465325e-13
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]

pd.DataFrame({'vif': vif[1:]}, index=X.columns).sort_values(by='vif', ascending=False)
%matplotlib inline

%config InlineBackend.figure_format ='retina'

import seaborn as sns 

import matplotlib.pyplot as plt

import statsmodels.stats.api as sms

sns.set_style('darkgrid')

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)



def homoscedasticity_test(model):

    '''

    Function for testing the homoscedasticity of residuals in a linear regression model.

    It plots residuals and standardized residuals vs. fitted values and runs Breusch-Pagan and Goldfeld-Quandt tests.

    

    Args:

    * model - fitted OLS model from statsmodels

    '''

    fitted_vals = model.predict()

    resids = model.resid

    resids_standardized = model.get_influence().resid_studentized_internal



    fig, ax = plt.subplots(1,2)



    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})

    ax[0].set_title('Residuals vs Fitted', fontsize=16)

    ax[0].set(xlabel='Fitted Values', ylabel='Residuals')



    sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})

    ax[1].set_title('Scale-Location', fontsize=16)

    ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')



    bp_test = pd.DataFrame(sms.het_breuschpagan(resids, model.model.exog), 

                           columns=['value'],

                           index=['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'])



    gq_test = pd.DataFrame(sms.het_goldfeldquandt(resids, model.model.exog)[:-1],

                           columns=['value'],

                           index=['F statistic', 'p-value'])



    print('\n Breusch-Pagan test ----')

    print(bp_test)

    print('\n Goldfeld-Quandt test ----')

    print(gq_test)

    print('\n Residuals plots ----')



homoscedasticity_test(lin_reg)
import statsmodels.tsa.api as smt



acf = smt.graphics.plot_acf(lin_reg.resid, lags=40 , alpha=0.05)

acf.show()
#Load Dataset from sklearn

from sklearn.datasets import load_boston





# Load Data

boston = load_boston()



# Data is in dictionary, Populate dataframe with data key

df = pd.DataFrame(boston.data)





# Columns are indexed, Fill in Column names with feature_names key

df.columns = boston.feature_names



# We need Median Value! boston.data contains only the features, no price value.



df['MEDV'] = pd.DataFrame(boston.target)

# First replace the 0 values with np.nan values

df.replace(0, np.nan, inplace=True)

# Check what percentage of each column's data is missing

df.isnull().sum()/len(df)
# Drop ZN and CHAS with too many missing columns

df = df.drop('ZN', axis=1)

df = df.drop('CHAS', axis=1)
import seaborn as sns



# Steps to remove redundant values

mask = np.zeros_like(df.corr())

mask[np.triu_indices_from(mask)] = True



# How to remove redundant correlation

# <https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib>



sns.set(rc={'figure.figsize': (8.5,8.5)})

sns.heatmap(df.corr().round(2), square=True, cmap='YlGnBu', annot=True, mask=mask);



# vmax emphasizes a color based on the gradient that you chose

# cmap is the color scheme of the heatmap

# square shapes the heatmap to a square for neatness

# annot shows the individual correlations of each pair of values

# mask removes redundacy and prevents repeat of the correlation values
# drop correlated values

columns = ['TAX', 'RAD', 'NOX', 'INDUS', 'DIS']

df = df.drop(columns=columns)
# Create multiple plots

features = df.drop('MEDV', 1).columns

target = df['MEDV']

plt.figure(figsize=(20,20))

for index, feature_name in enumerate(features):

    # 4 rows of plots, 13/3 == 4 plots per row, index+1 where the plot begins

    plt.subplot(4,len(features)/2, index+1)

    plt.scatter(df[feature_name], target)

    plt.title(feature_name, fontsize=15)

    plt.xlabel(feature_name, fontsize=8) #Removed for easier view of plots

    plt.ylabel('MEDV', fontsize=15)
import numpy as np

df["LOGLSTAT"] = df["LSTAT"].apply(np.log)

plt.figure(figsize=(20,10))



# showing plot 1

plt.subplot(1,2,1)

plt.scatter(df["LSTAT"], df['MEDV'], color='green')

plt.title('Original % Status of Neighborhood vs Median Price of House', fontsize= 20)

plt.xlabel('LSTAT',fontsize=20);

plt.ylabel('MEDV',fontsize=20);

plt.plot([0,40],[30,0])



# showing plot 2

plt.subplot(1,2,2)

plt.scatter(df["LOGLSTAT"], df['MEDV'], color='red')

plt.title('Log Transformed % Status of Neighborhood vs Median Price of House', fontsize= 20)

plt.xlabel('Transformed LSTAT',fontsize=20);

plt.ylabel('Transformed MEDV',fontsize=20);

plt.plot([0,4],[50,0])





#Apply global parameters

plt.rc('xtick', labelsize=20)

plt.rc('ytick', labelsize=20)



plt.show()
X = df[['LOGLSTAT', 'RM']]

y = df.MEDV

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 10)
from sklearn.linear_model import LinearRegression

# Create LinearRegression Instance

lrm = LinearRegression()



# Fit data on to the model

lrm.fit(X_train, y_train)



# Predict

y_predicted = lrm.predict(X_test)
from sklearn.metrics import mean_squared_error

def linear_metrics():

    r2 = lrm.score(X_test, y_test)

    rmse = (np.sqrt(mean_squared_error(y_test, y_predicted)))

    print('r-squared: {}'.format(r2))

    print('---------------------------------------')

    print('root mean squared error: {}'.format(rmse))

linear_metrics()
# Plot my predictions vs actual



plt.figure(figsize=(10,8))

plt.scatter(y_predicted, y_test)

plt.plot([0, 50], [0, 50], '--k')

plt.axis('tight')

plt.ylabel('The True Prices', fontsize=20);





plt.xlabel('My Predicted Prices', fontsize=20);

plt.title("Predicted Boston Housing Prices vs. Actual in $1000's", fontsize=20)



plt.show()
#Create Residuals Function



def residuals():

    plt.style.use('fivethirtyeight')

    plt.figure(figsize=(6,4))

    plt.scatter(lrm.predict(X_test), lrm.predict(X_test) - y_test, s=30)

    plt.xlabel('')

    plt.title('Residual Errors of Test Data')

    plt.hlines(0, xmin=-10, xmax=50, linewidth=1);



# The closer to 1, the more perfect the prediction

print('Variance score: {}'.format(lrm.score(X_test, y_test)))





residuals()
# Combine coefficient with their value

coeff = list(zip(X, lrm.coef_))



# sort keys by value of coefficient

print(['y-intercept = {}'.format(lrm.intercept_), sorted(coeff, key = lambda x: x[1])])
import statsmodels.tsa.api as smt



acf = smt.graphics.plot_acf(lin_reg.resid, lags=40 , alpha=0.05)

acf.show()