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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
#from datetime import datetime
from scipy.signal import butter, deconvolve
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from numpy import array
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import seaborn as sns
import lightgbm as lgb
state=pd.read_csv('/kaggle/input/corona/covid_19_india.csv')
state['Date'] = state['Date'].astype('datetime64[ns]')
state["date"]=state["Date"].dt.date

confirms=state.groupby(["date"])["Confirmed"].sum().reset_index()
deads=state.groupby(["date"])["Deaths"].sum().reset_index()
cures=state.groupby(["date"])["Cured"].sum().reset_index()

def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
	# define model model
	history = array(history)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
	# fit model
	model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = exp_smoothing_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
							models.append(cfg)
	return models

data = np.array(confirms["Confirmed"])
print(data)
   	# data split
n_test = 7
   	# model configs
cfg_list = exp_smoothing_configs()
   	# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
   	# list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)
    
print(scores)
def exp_smoothing_forecasting(data, config,n):
	t,d,s,p,b,r = config
	# define model model
	history = array(data)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
	# fit model
	model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
	# make one step forecast
	y_pred = model_fit.predict(len(data), len(data)+n-1)
	return y_pred

y_pred_7=exp_smoothing_forecasting(confirms["Confirmed"],['mul', True, None, None, False, False],7)

import datetime
start_date = confirms['date'].max()
prediction_dates = []
for i in range(7):
    date = start_date + datetime.timedelta(days=1)
    prediction_dates.append(date)
    start_date = date

for i in range(7):
    print(prediction_dates[i], y_pred_7[i])    
fig=plt.figure(figsize= (15,10))
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Predicted Values for the next 7 Days" , fontsize = 20)
plt.plot_date(y= y_pred_7,x= prediction_dates,linestyle ='dashed',color = '#ff9999',label = 'Predicted')
plt.plot_date(y=confirms['Confirmed'],x=confirms['date'],linestyle = '-',color = 'blue',label = 'Actual')
fig.autofmt_xdate()
data = np.array(confirms["Confirmed"])
print(data)
   	# data split
n_test = 14
   	# model configs
cfg_list = exp_smoothing_configs()
   	# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
   	# list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)
    
print(scores)
y_pred_14=exp_smoothing_forecasting(confirms["Confirmed"],['mul', True, None, None, False, False],14)
import datetime
start_date = confirms['date'].max()
prediction_dates = []
for i in range(14):
    date = start_date + datetime.timedelta(days=1)
    prediction_dates.append(date)
    start_date = date
for i in range(14):
    print(prediction_dates[i], y_pred_14[i])
fig=plt.figure(figsize= (15,10))
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Predicted Values for the next 14 Days" , fontsize = 20)
plt.plot_date(y= y_pred_14,x= prediction_dates,linestyle ='dashed',color = '#ff9999',label = 'Predicted')
plt.plot_date(y=confirms['Confirmed'],x=confirms['date'],linestyle = ':',color = 'blue',label = 'Actual')
plt.legend()
fig.autofmt_xdate()
autocorrelation_plot(confirms["Confirmed"])
pyplot.show()
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.7)
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
 
# load dataset

series = confirms["Confirmed"]
# evaluate parameters
p_values = [1,2,3,4,5,6,7,8,9,10,11]
d_values = range(0, 7)
q_values = range(0, 7)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)

#we will predict using the predictions 
def evaluate_arima_model(X, arima_order,n):
	# prepare training dataset
	#train_size = int(len(X) * 0.7)
	train = X
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(n):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(predictions[t])
	# calculate out of sample error
	
	return predictions
pred=evaluate_arima_model(confirms["Confirmed"].values,[1,2,0],7)


import datetime
start_date = confirms['date'].max()
prediction_dates = []
for i in range(7):
    date = start_date + datetime.timedelta(days=1)
    prediction_dates.append(date)
    start_date = date
for i in range(7):
    print(prediction_dates[i], pred[i])
fig=plt.figure(figsize= (15,10))
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Predicted Values for the next 7 Days using Arima" , fontsize = 20)
plt.plot_date(y=pred,x= prediction_dates,linestyle ='dashed',color = '#ff9999',label = 'Predicted')
plt.plot_date(y=confirms['Confirmed'],x=confirms['date'],linestyle = '-',color = 'blue',label = 'Actual')
fig.autofmt_xdate()

pred_14=evaluate_arima_model(confirms["Confirmed"].values,[1,2,0],14)
import datetime
start_date = confirms['date'].max()
prediction_dates = []
for i in range(14):
    date = start_date + datetime.timedelta(days=1)
    prediction_dates.append(date)
    start_date = date
for i in range(14):
    print(prediction_dates[i], pred_14[i])
fig=plt.figure(figsize= (15,10))
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Predicted Values for the next 7 Days using Arima" , fontsize = 20)
plt.plot_date(y=pred_14,x= prediction_dates,linestyle ='dashed',color = '#ff9999',label = 'Predicted')
plt.plot_date(y=confirms['Confirmed'],x=confirms['date'],linestyle = '-',color = 'blue',label = 'Actual')
fig.autofmt_xdate()

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
t=np.array(confirms["Confirmed"])
t=np.reshape(t,(-1,1))
X=sc.fit_transform(t)
pred=[]
for i in range(21):
    X_train = []
    X_test=[]
    y_train = []
    for i in range(7, len(X)):
        X_train.append(X[i-7:i, 0])
        y_train.append(X[i, 0])
    
    X_test.append(X[len(X)-7:len(X),0])
    X_test=np.array(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    #y_train=np.reshape(y_train,(len(y_train),1))
     
    
    #X_test=X_train[-1]
    #X_train=np.delete(X_train,-1,axis=0)
    #y_train=np.delete(y_train,-1,axis=0)
    #X_test=np.reshape(X_test,(1,7))
# Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    
    # Initialising the RNN
    regressor = Sequential()
    
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.1))
    
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.1))
    
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.1))
    
    # Adding the output layer
    regressor.add(Dense(units = 1))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 25)
    p=regressor.predict(X_test)
    
    X=np.append(X,p)
    X=np.reshape(X,(-1,1))
    p=sc.inverse_transform(p)
    pred.append(p)
    
    