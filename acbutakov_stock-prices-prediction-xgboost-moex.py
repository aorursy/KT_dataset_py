import pandas as pd

import numpy as np

import pandas_datareader.data as web

import matplotlib.pyplot as plt

import plotly as py

import plotly.graph_objs as go



from datetime import datetime

from xgboost import XGBRegressor

from xgboost import plot_importance



pd.set_option('display.max_rows', 10)

pd.set_option('display.max_columns', 100)



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))



def loadStockTable(Stock, StartDate, EndDate):

    MR = web.MoexReader(Stock, start=StartDate, end=EndDate)

    StockTable = MR.read()



    StockTable = StockTable[StockTable.BOARDID == 'TQBR']

    return StockTable



def plotStockTable(StockTable, title, filename, PredictTable = pd.DataFrame()):

    trace_high = go.Scatter(

    x=StockTable.index,

    y=StockTable.HIGH,

    name = "High",

    line = dict(color = '#17BECF'),

    opacity = 0.8)



    trace_low = go.Scatter(

        x=StockTable.index,

        y=StockTable.LOW,

        name = "Low",

        line = dict(color = '#7F7F7F'),

        opacity = 0.8)



    trace_open = go.Scatter(

        x=StockTable.index,

        y=StockTable.OPEN,

        name = "Open",

        line = dict(color = '#CD3333'),

        opacity = 0.8)



    trace_close = go.Scatter(

        x=StockTable.index,

        y=StockTable.CLOSE,

        name = "Close",

        line = dict(color = '#66CD00'),

        opacity = 0.8)



    if(len(PredictTable.index) > 0):

        trace_predict = go.Scatter(

            x=PredictTable.index,

            y=PredictTable.PREDICT,

            name = "Predict",

            line = dict(color = 'red'),

            opacity = 0.8)

    

    if(len(PredictTable.index) > 0):

        data = [trace_high, trace_low, trace_open, trace_close, trace_predict]

    else:

        data = [trace_high, trace_low, trace_open, trace_close]

        

    layout = dict(

        title=title,

        xaxis=dict(

            rangeselector=dict(

                buttons=list([

                    dict(count=7,

                         label='1w',

                         step='day',

                         stepmode='backward'),

                    dict(count=14,

                         label='2w',

                         step='day',

                         stepmode='backward'),

                    dict(count=1,

                         label='1m',

                         step='month',

                         stepmode='backward'),

                    dict(count=3,

                         label='3m',

                         step='month',

                         stepmode='backward'),

                    dict(count=6,

                         label='6m',

                         step='month',

                         stepmode='backward'),

                    dict(count=1,

                         label='1y',

                         step='year',

                         stepmode='backward'),

                    dict(count=3,

                         label='3y',

                         step='year',

                         stepmode='backward'),

                    dict(step='all')

                ])

            ),

            rangeslider=dict(

                visible = True

            ),

            type='date'

        )

    )



    fig = dict(data=data, layout=layout)

    py.offline.plot(fig, filename=filename)
#StockTable = loadStockTable('GAZP', '2010-01-01', '2019-05-15')

StockTable = pd.read_pickle("../input/StockTable.pkl")



plotStockTable(StockTable, 'АО "Газпром"', 'AO Gazprom.html')

StockTable
X = StockTable[['NUMTRADES', 'VALUE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME']]



X.NUMTRADES = X.NUMTRADES.astype('float64')

X.VOLUME    = X.VOLUME.astype('float64')



X.columns = ['NUMTRADES_1', 'VALUE_1', 'OPEN_1', 'LOW_1', 'HIGH_1', 'CLOSE_1', 'VOLUME_1']

X = X.reset_index()



Lags = np.arange(90) + 2

X_Lag = X[['NUMTRADES_1', 'VALUE_1', 'OPEN_1', 'LOW_1', 'HIGH_1', 'CLOSE_1', 'VOLUME_1']]



for i in Lags:

    X_Lag.loc[-1] = [np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]

    X_Lag.index = X_Lag.index + 1

    X_Lag = X_Lag.sort_index()

    X_Lag = X_Lag[:-1]



    X_Lag.columns = ['NUMTRADES_' + str(i), 'VALUE_' + str(i), 'OPEN_' + str(i), 'LOW_' + str(i), 'HIGH_' + str(i), 'CLOSE_' + str(i), 'VOLUME_' + str(i)]

    X = X.join(X_Lag, how='left')



X['MONTH'] = pd.to_datetime(X.TRADEDATE, format='%Y-%m-%d')

X['MONTH'] = X['MONTH'].dt.month



X['SEASON'] = 0

X.loc[(X['MONTH'] == 12)|(X['MONTH'] ==  1)|(X['MONTH'] ==  2), ['SEASON']] = 1

X.loc[(X['MONTH'] ==  3)|(X['MONTH'] ==  4)|(X['MONTH'] ==  5), ['SEASON']] = 2

X.loc[(X['MONTH'] ==  6)|(X['MONTH'] ==  7)|(X['MONTH'] ==  8), ['SEASON']] = 3

X.loc[(X['MONTH'] ==  9)|(X['MONTH'] == 10)|(X['MONTH'] == 11), ['SEASON']] = 4



X = X[X.index >= 90]

X
#train_columns = X.columns

train_columns = [col for col in X.columns if 'CLOSE' in col]



X_train = X[train_columns]

#X_train = X_train.drop(columns={'TRADEDATE', 'NUMTRADES_1', 'VALUE_1', 'OPEN_1', 'LOW_1', 'HIGH_1', 'CLOSE_1', 'VOLUME_1'})

X_train = X_train.drop('CLOSE_1', axis=1)

Y_train = X[['CLOSE_1']]#X[['NUMTRADES_1', 'VALUE_1', 'OPEN_1', 'LOW_1', 'HIGH_1', 'CLOSE_1', 'VOLUME_1']]



X_train = X_train.reset_index(drop=True)

Y_train = Y_train.reset_index(drop=True)



ValidSize = 1



X_valid = X_train[X_train.index >= (len(X_train.index) - ValidSize)]

Y_valid = Y_train[Y_train.index >= (len(Y_train.index) - ValidSize)]



X_train = X_train[:-ValidSize]

Y_train = Y_train[:-ValidSize]



model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)
plot_features(model, (10,30))
XBuf = X[X.columns]

XBuf = XBuf.reset_index(drop=True)



X_pred = XBuf[train_columns]

#X_pred = X_pred.drop(columns={'TRADEDATE', 'NUMTRADES_1', 'VALUE_1', 'OPEN_1', 'LOW_1', 'HIGH_1', 'CLOSE_1', 'VOLUME_1'})

X_pred = X_pred.drop('CLOSE_1', axis=1)



Y_pred = pd.DataFrame(model.predict(X_pred), columns={'PREDICT'})

Y_pred = Y_pred.join(XBuf, how='left')



Y_pred.index = Y_pred.TRADEDATE

Y_pred



plotStockTable(StockTable, 'АО "Газпром"', 'AO Gazprom_predict.html', Y_pred)
X_valid