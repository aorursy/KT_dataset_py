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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse
import xlrd


from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',3000)
# importing the data in a dataframe

crime_rate_df = pd.read_excel('/kaggle/input/population-data1/population_data.xls')

crime_rate_df.info()
crime_rate_df.head()
crime_rate_df['interact_variables'] = crime_rate_df['2015 total_pop']*crime_rate_df['2015 homocide_intent']
crime_rate_df['interact_variables2'] = crime_rate_df['2015 prct_female_avd_edu']*crime_rate_df['2015 pop_adv_edu']

# Y is our target variable
Y = crime_rate_df['2016 homocide_intent']
# X is our feature set (independen variable)
X = crime_rate_df[['interact_variables2','interact_variables','2015 pop_adv_edu','2015 per_pop_basic_edu','2015 prct_unemploy rate','2014 homocide_intent']]
lrm = linear_model.LinearRegression()

lrm.fit(X,Y)

print('n\Coefficients: \n', lrm.coef_)
print('n\Intercept: \n', lrm.intercept_)
X = sm.add_constant(X)

results = sm.OLS(Y, X).fit()
results.summary()
# Y is our target variable 
Y1 =  crime_rate_df['2017 homocide_intent']
# X is our feature set (independent variable)
X1 = crime_rate_df[['2011 homocide_intent','2012 homocide_intent','2013 homocide_intent','2014 homocide_intent','2015 homocide_intent']]

lrm = linear_model.LinearRegression()

lrm.fit(X1,Y1)

# Inspect the results.
print('\nCoefficients: \n', lrm.coef_)
print('\nIntercept: \n',lrm.intercept_)
X1 = sm.add_constant(X1)

results = sm.OLS(Y1, X1).fit()
results.summary()
# Y is our target variable 
Y1_no_constant =  crime_rate_df['2017 homocide_intent']
# X is our feature set (independent variable)
X1_no_constant = crime_rate_df[['2011 homocide_intent','2012 homocide_intent','2013 homocide_intent','2014 homocide_intent','2015 homocide_intent']]

lrm = linear_model.LinearRegression()

lrm.fit(X1_no_constant,Y1_no_constant)

# Inspect the results.
print('\nCoefficients: \n', lrm.coef_)
print('\nIntercept: \n',lrm.intercept_)
results = sm.OLS(Y1_no_constant, X1_no_constant).fit()
results.summary()
# Y is our target variable 
Y2 =  crime_rate_df['2016 homocide_intent']
# X is our feature set (independent variable)
X2 = crime_rate_df[['2011 homocide_intent','2012 homocide_intent','2013 homocide_intent','2014 homocide_intent','2015 homocide_intent']]

lrm = linear_model.LinearRegression()

lrm.fit(X2,Y2)

# Inspect the results.
print('\nCoefficients: \n', lrm.coef_)
print('\nIntercept: \n',lrm.intercept_)
X2 = sm.add_constant(X2)

results = sm.OLS(Y2, X2).fit()
results.summary()
# Y is our target variable 
Y2_no_constant =  crime_rate_df['2016 homocide_intent']
# X is our feature set (independent variable)
X2_no_constant = crime_rate_df[['2011 homocide_intent','2012 homocide_intent','2013 homocide_intent','2014 homocide_intent','2015 homocide_intent']]

lrm = linear_model.LinearRegression()

lrm.fit(X2_no_constant,Y2_no_constant)

# Inspect the results.
print('\nCoefficients: \n', lrm.coef_)
print('\nIntercept: \n',lrm.intercept_)
results = sm.OLS(Y2_no_constant, X2_no_constant).fit()
results.summary()
X1_train_no_constant, X1_test_no_constant, Y1_train_no_constant, Y1_test_no_constant = train_test_split(X1_no_constant,Y1_no_constant, test_size = 0.30, random_state=1450)

print("The number of observations in training set is {}".format(X1_train_no_constant.shape[0]))
print("The number of observations in the test set is {}".format(X1_test_no_constant.shape[0]))
results1_no_constant = sm.OLS(Y1_train_no_constant,X1_train_no_constant).fit()

results1_no_constant.summary()
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, test_size = 0.30, random_state = 1450) 

print("The number of observations in training set is {}".format(X1_train.shape[0]))
print("The number of observations in the test set is {}".format(X1_test.shape[0]))
results1 = sm.OLS(Y1_train, X1_train).fit()
results1.summary()
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.30, random_state = 1450)
print("The number of observations in the training set is: {}".format(X2_train.shape[0]))
print("The number of observations in the test set is: {}".format(X2_test.shape[0]))
results2 = sm.OLS(Y2_train, X2_train).fit()
results2.summary()
X2_train_no_constant, X2_test_no_constant, Y2_train_no_constant, Y2_test_no_constant = train_test_split(X2_no_constant, Y2_no_constant, test_size = 0.30, random_state = 1450)
print("The number of observations in the training set is: {}".format(X2_train_no_constant.shape[0]))
print("The number of observations in the test set is: {}".format(X2_test_no_constant.shape[0]))
results2_no_constant = sm.OLS(Y2_train_no_constant, X2_train_no_constant).fit()
results2_no_constant.summary()
Y1_preds = results1.predict(X1_test)

plt.scatter(Y1_test, Y1_preds)
plt.plot(Y1_test, Y1_test, color='red')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('2017 Homocide intent predictions')
plt.show()

print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(Y1_test, Y1_preds)))
print("Mean squared error of the prediction is: {}".format(mse(Y1_test, Y1_preds)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y1_test, Y1_preds)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y1_test - Y1_preds) / Y1_test)) * 100))

Y1_no_constant_preds = results1_no_constant.predict(X1_test_no_constant)

plt.scatter(Y1_test_no_constant, Y1_no_constant_preds)
plt.plot(Y1_test_no_constant, Y1_test_no_constant, color='red')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('2017 Homocide intent predictions (no constant)')
plt.show()

print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(Y1_test_no_constant, Y1_no_constant_preds)))
print("Mean squared error of the prediction is: {}".format(mse(Y1_test_no_constant, Y1_no_constant_preds)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y1_test_no_constant, Y1_no_constant_preds)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y1_test_no_constant - Y1_no_constant_preds) / Y1_no_constant_preds)) * 100))

Y2_no_constant_preds = results2_no_constant.predict(X2_test_no_constant)

plt.scatter(Y2_test_no_constant, Y2_no_constant_preds)
plt.plot(Y2_test_no_constant, Y2_test_no_constant, color='red')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('2016 Homocide intent predictions')
plt.show()

print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(Y2_test_no_constant, Y2_no_constant_preds)))
print("Mean squared error of the prediction is: {}".format(mse(Y2_test_no_constant, Y2_no_constant_preds)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y2_test_no_constant, Y2_no_constant_preds)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y2_test_no_constant - Y2_no_constant_preds) / Y2_no_constant_preds)) * 100))

Y2_constant_preds = results2.predict(X2_test)

plt.scatter(Y2_test, Y2_constant_preds)
plt.plot(Y2_test, Y2_test, color='red')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('2016 Homocide intent predictions')
plt.show()

print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(Y2_test, Y2_constant_preds)))
print("Mean squared error of the prediction is: {}".format(mse(Y2_test, Y2_constant_preds)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y2_test, Y2_constant_preds)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y2_test - Y2_constant_preds) / Y2_constant_preds)) * 100))

