import gc

import numpy as np

import pandas as pd

from pylab import rcParams

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARIMA

from scipy.stats import probplot



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")
column_types = {

    'store':'int8',

    'item':'int8',

    'sales':'float64',

}

train = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv',dtype=column_types,parse_dates=['date'])

test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv',dtype=column_types,parse_dates=['date'])

submission = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')
train.info()
test.info()
print('Train from {%s} to {%s}' % (train.date.min(),train.date.max()))

print('Test from {%s} to {%s}' % (test.date.min(),test.date.max()))
pre_skew = train['sales'].skew()

pre_kurt = train['sales'].kurt()



train['sales'] = np.log1p(train['sales'])



print('Training Set Sales Skew from {%f} downto {%f}' % (pre_skew,train['sales'].skew()))

print('Training Set Sales Kurtosis from {%f} downto {%f}' % (pre_kurt,train['sales'].kurt()))
sales = train[train.store==1][train.item==1].sales

rcParams['figure.figsize'] = 20, 5

plt.plot(sales)
rcParams['figure.figsize'] = 20, 10

figure = sm.tsa.seasonal_decompose(sales,freq=360).plot() # decompose with 360

figure.show()
sales_diff1 = sales.diff(1).iloc[1:]

rcParams['figure.figsize'] = 20, 5

plt.plot(sales_diff1)
sales_diff2 = sales.diff(2).iloc[2:]

rcParams['figure.figsize'] = 20, 5

plt.plot(sales_diff2)
from statsmodels.graphics.tsaplots import plot_acf

rcParams['figure.figsize'] = 20, 5

figure = plot_acf(sales_diff1,lags=30,title='Train sales autocorrelation in 30 lags')

figure.show()
from statsmodels.graphics.tsaplots import plot_pacf

rcParams['figure.figsize'] = 20, 5

figure = plot_pacf(sales_diff1,lags=30,title='Train sales partial autocorrelation in 30 lags')

figure.show()
def evaluate_arima_model(X, arima_order):

    # prepare training dataset

    train_size = int(len(X) * 0.8)

    train, test = X[0:train_size], X[train_size:]

    history = [x for x in train]

    # make predictions

    predictions = list()

    for t in range(len(test)):

        model = ARIMA(history, order=arima_order)

        model_fit = model.fit(disp=0)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(test[t])

    # calculate out of sample error

    error = mean_squared_error(test, predictions)

    return error

    

# evaluate combinations of p, d and q values for an ARIMA model

def evaluate_models(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float32')

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    mse = evaluate_arima_model(dataset, order)

                    if mse < best_score:

                        best_score, best_cfg = mse, order

                    print('ARIMA%s MSE=%.3f' % (order,mse))

                except:

                    continue

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

    return best_cfg, best_score
# p_values = range(6,9)

# d_values = range(0,3)

# q_values = range(0,3)

# best_cfg, best_score = evaluate_models(sales.values, p_values, d_values, q_values)

best_cfg=(7,1,1) # for sales
rcParams['figure.figsize'] = 20, 5

model = ARIMA(sales, order=best_cfg) # 7 from ar, 1 from diff, 1 for ma.

result = model.fit()

print(result.summary())

result.plot_predict(start=750, end=1000)

plt.show()
rmse = np.sqrt(mean_squared_error(sales.diff().iloc[1:1001].values, result.predict(start=1,end=1000)))

print("The root mean squared error with ARIMA(1,1,0) is {%f}." % rmse)
y = []

y_hat = []

models = {}

for k,si in train[train.store<=3].groupby(['store','item']):

    print(k,si.sales.min(),si.sales.max())

#     y+=(si.sales.diff(1).tolist()[1:]) # 对应order中的d=1

    model = ARIMA(si.sales.values, order=best_cfg)

    result = model.fit()

#     y_hat+=(result.predict(start=1).tolist())

    models[str(si.store)+'_'+str(si.item)] = result
for k,si in train[train.store>3][train.store<=6].groupby(['store','item']):

    print(k,si.sales.min(),si.sales.max())

#     y+=(si.sales.diff(1).tolist()[1:]) # 对应order中的d=1

    model = ARIMA(si.sales.values, order=best_cfg)

    result = model.fit()

#     y_hat+=(result.predict(start=1).tolist())

    models[str(si.store)+'_'+str(si.item)] = result
for k,si in train[train.store>6].groupby(['store','item']):

    print(k,si.sales.min(),si.sales.max())

#     y+=(si.sales.diff(1).tolist()[1:]) # 对应order中的d=1

    model = ARIMA(si.sales.values, order=best_cfg)

    result = model.fit()

#     y_hat+=(result.predict(start=1).tolist())

    models[str(si.store)+'_'+str(si.item)] = result
# rcParams['figure.figsize'] = 20, 5

# plt.plot(y[:200])

# plt.plot(y_hat[:200])

# plt.show()
# rmse = np.sqrt(mean_squared_error(y, y_hat))

# print("The root mean squared error with ARIMA{%s} between y and y_hat is {%f}." % (best_cfg,rmse))
models[str(1)+'_'+str(1)].predict(start='2018-01-01 00:00:00')
1/0
for row in test.itertuples():

    print(getattr(row,'Index'), getattr(row,'date'), getattr(row,'store'), getattr(row,'item'))

    models[str(getattr(row,'store'))+'_'+str(getattr(row,'item'))].predict(start=getattr(row,'date'))