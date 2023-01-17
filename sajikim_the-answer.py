# importing the libraries.

import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima_model import ARIMA

from sklearn.preprocessing import StandardScaler

from math import sqrt

import itertools



import warnings

warnings.filterwarnings('ignore')
!pip install pmdarima
def predict_arima_model(train, period, order, maxlags=8, ic='aic'):

    # Feature Scaling

    stdsc = StandardScaler()

    train_std = stdsc.fit_transform(train.values.reshape(-1, 1))

    # fit model

    model = ARIMA(train_std, order=order)

    model_fit = model.fit(maxlags=maxlags, ic=ic, disp=0)

    # make prediction

    yhat = model_fit.predict(len(train), len(train) + period -1, typ='levels')

    # inverse transform

    yhat = stdsc.inverse_transform(np.array(yhat).flatten())

    return yhat

    

def make_future_dates(last_date, period):

    prediction_dates=pd.date_range(last_date, periods=period+1, freq='B')

    return prediction_dates[1:]



import pmdarima

def evaluate_arima(df):

    best_model = pmdarima.auto_arima(df,                                    

                                     seasonal=False, stationary=False, 

                                     m=7, information_criterion='aic', 

                                     max_order=20,                                     

                                     max_p=10, max_d=2, max_q=10,                                     

                                     max_P=10, max_D=2, max_Q=10,                                   

                                     error_action='ignore')

    print("best model --> (p, d, q):", best_model.order)
# prepare dataset

data_org = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv',header=0,parse_dates=[0])

data=data_org[["Date","Price"]].tail(101)

data.set_index('Date',drop=True,inplace=True)



# make diff

data['Price'] = data['Price'].diff()

data=data[1:]



#evaluate_arima(data)



# predict future period with best parameter

forecast_out = 34

future_dates = make_future_dates(data.index[-1], forecast_out)

predictions = predict_arima_model(data,len(future_dates),(2, 1, 1))

submission = pd.DataFrame({'Price':predictions},index=future_dates)



# invert diff

base = data_org.tail(1)["Price"]

for index, row in submission.iterrows():

    base = base + row["Price"]

    submission.at[index, 'Price'] = base



# submission file

submission.index.name = "Date"

submission.to_csv("submission.csv", index=True)



submission