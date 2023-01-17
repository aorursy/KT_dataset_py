# importing the libraries.
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
import itertools

import warnings
warnings.filterwarnings('ignore')
!pip install pmdarima
# common function
def evaluate_arima_model(train, test, order, maxlags=8, ic='aic'):
    # prepare training dataset
    history = [x for x in train]
    # make predictions
    predictions = list()
    # rolling forecasts
    for t in range(len(test)):
        # predict
        model = ARIMA(history, order=order)
        model_fit = model.fit(maxlags=maxlags, ic=ic, disp=0)
        yhat = model_fit.forecast()[0]
        # invert transformed prediction
        predictions.append(yhat)
        # observation
        history.append(test[t])
    # calculate mse
    mse = mean_squared_error(test, predictions)
    return predictions, mse

def evaluate_arima_models(train, test, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    pdq = list(itertools.product(p_values, d_values, q_values))
    for order in pdq:
        try:
            predictions, mse = evaluate_arima_model(train, test, order)
            if mse < best_score:
                best_score, best_cfg = mse, order
            print('Model(%s) mse=%.3f' % (order,mse))
        except:
            continue
    print('Best Model(%s) mse=%.3f' % (best_cfg, best_score)) 
    return best_cfg

def get_data_from_EIA_local():
    df = pd.read_csv("../input/cushing-ok-wti-spot-price-fob/Cushing_OK_WTI_Spot_Price_FOB_20200626.csv", header=4, parse_dates = [0])
    df.columns=["Date", "Price"]
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df
# prepare dataset
df_org=get_data_from_EIA_local()
data=df_org['2019-01-01':].copy()
data.Price["2020-04-20"]=(data.Price["2020-04-17"] + data.Price["2020-04-21"]) / 2
split = int(0.80*len(data))
train_data, test_data = data[0:split], data[split:]
# evaluate by auto_arima
import pmdarima
import time

start = time.time()
best_model = pmdarima.auto_arima(train_data['Price'],                                    
                                 seasonal=False, stationary=False, 
                                 m=7, information_criterion='aic', 
                                 max_order=20,                                     
                                 max_p=10, max_d=2, max_q=10,                                     
                                 max_P=10, max_D=2, max_Q=10,                                   
                                 error_action='ignore')
print("best model --> (p, d, q):", best_model.order)
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
# evaluate parameters
p_values = range(1, 10)
d_values = range(1, 2)
q_values = range(1, 10)
start = time.time()
evaluate_arima_models(train_data['Price'], test_data['Price'], p_values, d_values, q_values)
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")