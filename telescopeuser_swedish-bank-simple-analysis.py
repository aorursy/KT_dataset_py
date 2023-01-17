import numpy as np

import pandas as pd

import re

import seaborn as sns

from sklearn import cross_validation, linear_model, metrics, ensemble

### R2 estimation

import statsmodels

import statsmodels.formula.api as smf

import statsmodels.stats.api as sms

from statsmodels.graphics.regressionplots import plot_leverage_resid2

###

%pylab inline



pd.options.mode.chained_assignment = None
raw = pd.read_csv('../input/Interestrate and inflation Sweden 1908-2001.csv')

raw.head()
raw.info()
# let's drop all NA

data = raw.dropna()

# price level uses comma to separate thousands, let's fix that

data['Price level'] = data['Price level'].apply(lambda x: re.sub(',', '', x))



cols = ['Period', 'Central bank interest rate diskonto average', 'Price level']

for col in cols:

    data[col] = pd.to_numeric(data[col], errors='coerce')

    

data.info()
# plot period/interest rate

sns.pairplot(data)
data.corr(method='pearson')
data.corr(method='spearman')
def time_series_cv(estimator, X, y, folds=5, metrics_f=metrics.mean_squared_error):

    '''

    Performs cross validation on ESTIMATOR and the data (X, y) using

    forward chaining.

    The score is the result of metrics_f(prediction, train_part_y)

    Here is the example of the data split to 6 folds

    for list [1 2 3 4 5 6]:

    TRAIN | TEST

    [1] | [2]

    [1 2] | [3]

    [1 2 3] | [4]

    [1 2 3 4] | [5]

    [1 2 3 4 5] | [6]

    '''

    assert X.shape[0] == y.shape[0], "Features and targets of different sizes {} != {}".format(\

    X.shape[0], y.shape[0]

    )



    results = []

    fold_size = int(np.ceil(X.shape[0] / folds))



    for i in range(1, folds):

        split = i*fold_size

        trainX, trainY = X[:split], y[:split]

        testX, testY = X[split:], y[split:]

        estimator.fit(trainX, trainY)

        predictions = estimator.predict(testX)

        results.append(metrics_f(predictions, testY))



    return np.array(results)
# formally, inflation <--> int.rate dependency is kind of a time series, so we have to use 

# "time-series" version of CV

regressor = linear_model.LinearRegression(fit_intercept=True)

report = time_series_cv(regressor, data[['Central bank interest rate diskonto average']],

                                                     data[['Inflation']], folds=5, metrics_f=metrics.mean_absolute_error)

print(report)

print('mean={}; std={}'.format(np.mean(report), np.std(report)))

data['Inflation'].plot(kind='hist', title='Inflation hist')
regressor.fit(data[['Central bank interest rate diskonto average']],

              data[['Inflation']])

print(regressor.coef_, regressor.intercept_)

plt.plot(data['Central bank interest rate diskonto average'],

         data['Inflation'], '*', label='original points')

w0, w1 = regressor.intercept_[0], regressor.coef_[0][0]

x = np.arange(0, 12, 0.2)

plt.plot(x, w0 + w1 * x, label='predicted curve')

plt.legend(loc=2)
data.rename(columns={'Central bank interest rate diskonto average': 'Interest'}, inplace=True)

data.head()
formula = 'Inflation ~ Interest'

model = smf.ols(formula, data=data)

fitted = model.fit()



print(fitted.summary())
print('Breusch-Pagan test: p=%f' % sms.het_breushpagan(fitted.resid, fitted.model.exog)[1])