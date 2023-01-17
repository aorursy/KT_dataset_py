import pandas as pd

import matplotlib.pyplot  as plt

 

def parser(x):

	return pd.datetime.strptime('190'+x, '%Y-%m')

 

series = pd.read_csv('../input/shampoo-saled-dataset/shampoo_sales.csv', 

                     header=0, parse_dates=[0], index_col=0, 

                     squeeze=True, date_parser=parser)

print(series.head())

series.plot()

plt.show()
from pandas.plotting import autocorrelation_plot



autocorrelation_plot(series)

plt.show()
# importing arima library

from statsmodels.tsa.arima_model import ARIMA





model = ARIMA(series, order=(5,1,0))

model_fit = model.fit(disp=0)

print(model_fit.summary())



# plot residual errors

residuals = pd.DataFrame(model_fit.resid)

residuals.plot()

plt.title('ARMA Fit Residual Error Line Plot')

plt.xlabel('Months')

plt.ylabel('Residual Error')

plt.show()
residuals.plot(kind='kde')

plt.title('ARMA Fit Residual Error Density Plot')

plt.xlabel('Residual Values')

plt.grid()

plt.show()

print(residuals.describe())
from sklearn.metrics import mean_squared_error

 



X = series.values

size = int(len(X) * 0.66)





train, test = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions = []





for t in range(len(test)):

	model = ARIMA(history, order=(5,1,0))

	model_fit = model.fit(disp=0)

	output = model_fit.forecast()

	yhat = output[0]

	predictions.append(yhat)

	obs = test[t]

	history.append(obs)

	print('predicted=%f, expected=%f' % (yhat, obs))

    









# plot

plt.plot(test, label = 'original sales', marker = '*')

plt.plot(predictions, color='red', label = 'predicted sales', marker = '*')

plt.title('Performance Evaluation')

plt.xlabel('Future Steps')

plt.ylabel('Sales')

plt.legend()

plt.show()
import math

error = mean_squared_error(test, predictions)

print('Test Root Mean Squared Error: %.3f' % math.sqrt(error))