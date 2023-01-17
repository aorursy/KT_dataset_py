import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from datetime import datetime

from sklearn.model_selection import train_test_split



import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARIMA



%matplotlib inline



# We are required to do this in order to avoid "FutureWarning" issues.

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
cov = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
cov.head()
cov.info()
cov.describe()
cov['Date']
cov_confirmed = cov.groupby('Date')[['ConfirmedCases']].sum()
cov_confirmed.tail()
# Code modified from code written by Matthew Garton.



def plot_series(cov_confirmed, cols=None, title='Title', xlab=None, ylab=None, steps=1):

    

    # Set figure size to be (18, 9).

    plt.figure(figsize=(18,9))

    

    # Iterate through each column name.

    for col in cols:

            

        # Generate a line plot of the column name.

        # You only have to specify Y, since our

        # index will be a datetime index.

        plt.plot(cov_confirmed[col])

        

    # Generate title and labels.

    plt.title(title, fontsize=26)

    plt.xlabel(xlab, fontsize=20)

    plt.ylabel(ylab, fontsize=20)

    

    # Enlarge tick marks.

    plt.yticks(fontsize=18)

    plt.xticks(cov_confirmed.index[0::steps], fontsize=18);
# Generate a time plot.

plot_series(cov_confirmed, ['ConfirmedCases'], title = 'Worldwide Count of Confirmed COVID-19 Cases', steps = 14)
# first 5 values of the COVID-19 series.

cov_confirmed['ConfirmedCases'][0:5]
cov_confirmed['ConfirmedCases'][:5].diff(1)
cov_confirmed['ConfirmedCases'][:5].diff(1).diff(1)
cov_confirmed['ConfirmedCases'][:5].diff(1).diff(1).diff(1)
cov_confirmed['first_diff_confirmed'] = cov_confirmed['ConfirmedCases'].diff(1)

cov_confirmed['second_diff_confirmed'] = cov_confirmed['ConfirmedCases'].diff(1).diff(1)

cov_confirmed['third_diff_confirmed'] = cov_confirmed['ConfirmedCases'].diff(1).diff(1).diff(1)

cov_confirmed.head()
# Examine confirmed cases, differenced once.

plot_series(cov_confirmed,

            ['first_diff_confirmed'],

            title = "Change in Confirmed Cases from Day to Day",

            steps=14)
# Examine confirmed cases, differenced twice.

plot_series(cov_confirmed,

            ['second_diff_confirmed'],

            title = "Change in Confirmed Cases from Day to Day",

            steps=14)
# Examine confirmed cases, differenced thrice.

plot_series(cov_confirmed,

            ['third_diff_confirmed'],

            title = "Change in Confirmed Cases from Day to Day",

            steps=14)
# Import Augmented Dickey-Fuller test.

from statsmodels.tsa.stattools import adfuller



# Run ADF test on original (non-differenced!) data.

adfuller(cov_confirmed['ConfirmedCases'])
# Code written by Joseph Nelson.



def interpret_dftest(dftest):

    dfoutput = pd.Series(dftest[0:2], index=['Test Statistic','p-value'])

    return dfoutput
# Run ADF test on original (non-differenced!) data.



interpret_dftest(adfuller(cov_confirmed['ConfirmedCases']))
# Run the ADF test on our once-differenced data.

interpret_dftest(adfuller(cov_confirmed['first_diff_confirmed'].dropna()))
# Run the ADF test on our twice-differenced data.

interpret_dftest(adfuller(cov_confirmed['second_diff_confirmed'].dropna()))
# Create train-test split.

y_train, y_test = train_test_split(cov_confirmed['second_diff_confirmed'], test_size=0.1, shuffle=False)
# Starting AIC, p, and q.

best_aic = 99 * (10 ** 16)

best_p = 0

best_q = 0

# Use nested for loop to iterate over values of p and q.

for p in range(5):

    for q in range(5):

        # Insert try and except statements.

        try:

            # Fitting an ARIMA(p, 2, q) model.

            print(f'Attempting to fit ARIMA({p}, 2, {q}).')

            # Instantiate ARIMA model.

            arima = ARIMA(endog = y_train.dropna(), # endog = Y variable

                          order = (p, 2, q)) # values of p, d, q

            # Fit ARIMA model.

            model = arima.fit()

            # Print out AIC for ARIMA(p, 2, q) model.

            print(f'The AIC for ARIMA({p},2,{q}) is: {model.aic}')

            # Is my current model's AIC better than our best_aic?

            if model.aic < best_aic:

                # If so, let's overwrite best_aic, best_p, and best_q.

                best_aic = model.aic

                best_p = p

                best_q = q

        except:

            pass

print()

print()

print('MODEL FINISHED!')

print(f'Our model that minimizes AIC on the training data is the ARIMA({best_p},2,{best_q}).')

print(f'This model has an AIC of {best_aic}.')
# Instantiate best model.

model = ARIMA(endog = y_train.dropna(),  # Y variable

              order = (4, 2, 2))

# Fit ARIMA model.

arima = model.fit()

# Generate predictions based on test set.

preds = model.predict(params = arima.params,

                      start = y_test.index[0],

                      end = y_test.index[-1])

# Plot data.

plt.figure(figsize=(12,8))

# Plot training data.

plt.plot(y_train.index, pd.DataFrame(y_train).diff(), color = 'blue')

# Plot testing data.

plt.plot(y_test.index, pd.DataFrame(y_test).diff(), color = 'orange')

# Plot predicted test values.

plt.plot(y_test.index, preds, color = 'green')

plt.title(label = 'Twice-Differenced Confirmed Cases with ARIMA(0, 2, 1) Predictions', fontsize=16)

plt.show();
cov_fatal = cov.groupby('Date')[['Fatalities']].sum()
cov_fatal.tail()
# Code modified from code written by Matthew Garton.



def plot_series(cov_fatal, cols=None, title='Title', xlab=None, ylab=None, steps=1):

    

    # Set figure size to be (18, 9).

    plt.figure(figsize=(18,9))

    

    # Iterate through each column name.

    for col in cols:

            

        # Generate a line plot of the column name.

        # You only have to specify Y, since our

        # index will be a datetime index.

        plt.plot(cov_fatal[col])

        

    # Generate title and labels.

    plt.title(title, fontsize=26)

    plt.xlabel(xlab, fontsize=20)

    plt.ylabel(ylab, fontsize=20)

    

    # Enlarge tick marks.

    plt.yticks(fontsize=18)

    plt.xticks(cov_confirmed.index[0::steps], fontsize=18);
# Generate a time plot.

plot_series(cov_fatal, ['Fatalities'], title = 'Worldwide Count of COVID-19 Fatalities', steps = 14)