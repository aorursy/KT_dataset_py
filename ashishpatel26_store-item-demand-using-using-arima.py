import warnings
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
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
#read in .csv data file
data_train = pd.read_csv("../input/train.csv", parse_dates = ["date"], index_col = "date")
#explore this data set
print (data_train.head())
print ("\n Data Types:")
print (data_train.dtypes)
data_train.index
m = max(data_train["store"])
n = max(data_train["item"])
k = data_train.shape[0]
p = int(k/(m*n))

#slice by different store and item
ts_train = data_train.iloc[0:p, 2]
#Determing rolling statistics
rolmean = pd.Series.rolling(ts_train, window=12).mean()
rolstd = pd.Series.rolling(ts_train, window=12).std()
#Plot rolling statistics:
orig = plt.plot(ts_train, color="blue",label="Original")
mean = plt.plot(rolmean, color="red", label="Rolling Mean")
std = plt.plot(rolstd, color="black", label = "Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean & Standard Deviation")
plt.show(block=False)
#Perform Dickey-Fuller test:
print ("Results of Dickey-Fuller Test:")
dftest = adfuller(ts_train, autolag="AIC")
dfoutput = pd.Series(dftest[0:4], index=["Test Statistic","p-value","#Lags Used","Number of Observations Used"])
for key,value in dftest[4].items():
    dfoutput["Critical Value (%s)"%key] = value
print (dfoutput)
warnings.filterwarnings("ignore")
evaluate_models(ts_train, [0,1,2], [0,1,2], [0,1,2])
