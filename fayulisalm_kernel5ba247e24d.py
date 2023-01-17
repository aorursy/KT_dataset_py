

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:11:32 2020

@author: Sam
"""


import warnings
from math import sqrt
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

#Evaluate an ARIMA model for a given order(p,d,q)

def evaluate_arima_model(X, arima_order):
    #prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    
    #making prediction
    
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit(disp = 0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
        
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse  

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE = %.3f' % (order, rmse))
                except:
                     continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


#load dataset

dateparse = lambda x:datetime.strptime(x, '%d/%m/%Y %H:%M')
series = pd.read_csv('../input/processeddatehourload/processed-date-hour-load.csv', header=0, index_col=0, parse_dates=True,
squeeze=True,  date_parser = dateparse)
series = series.asfreq(freq='H', method='ffill')
#series.index = pd.to_datetime(series.index).to_period('m')   

#evaluate parameters
p_values = [0,1,2,3,4,5]
d_values = [0,1,2]
q_values = [0,1,2,3,4,5]
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)