import pandas                 as pd

import numpy                  as np

#import pandas_datareader.data as web

import matplotlib.pyplot      as plt

import plotly                 as py

import plotly.graph_objs      as go

import threading              as td

#import talib                  as ta

import sklearn                as skl

import requests               as rq



from datetime                import datetime

from datetime                import timedelta

from xgboost                 import XGBRegressor

from xgboost                 import plot_importance

from threading               import Thread

from xml.etree.ElementTree   import fromstring, ElementTree



from sklearn.model_selection  import train_test_split

from sklearn.ensemble         import GradientBoostingRegressor

from sklearn.model_selection  import KFold, cross_val_score

from sklearn.model_selection  import GridSearchCV

from sklearn.externals.joblib import parallel_backend



pd.set_option('display.max_rows', 10)

pd.set_option('display.max_columns', 100)



StockTableColumnsForUse = pd.Series(['TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME'])

StockTableDateColumns   = pd.Series(['TRADEDATE', 'MONTH', 'SEASON'])



LagsNumb = 14

MaxWindowSize = 364

XColumnsWithoutLags = 0

XColumnsWithLags = 0



def cv_rmse(model, X, Y, cv):

    rmse = np.sqrt(-cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv=cv))

    return (rmse)



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



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



def getLagColumns(X):

    

    LagColumns = X.columns[1:]

    

    return LagColumns



def loadStockTable(Stock, StartDate, EndDate):    

    StockTable = pd.read_csv("../input/moex-gazp/" + Stock + '_S.csv')

    StockTable.TRADEDATE = pd.to_datetime(StockTable.TRADEDATE, format="%Y/%m/%d")

    

    return StockTable



def loadFutureTable(Future, StartDate, EndDate):

    FutureTable = pd.read_csv("../input/moex-gazp/" + Future + '_ready.csv')



    FutureTable = FutureTable.replace('-', 0) # в день начала торгов нового фьючерса будут прочерки в ячейках изменения к предыдущему дню



    for column in FutureTable.columns:

        FutureTable[column] = FutureTable[column].astype('str').str.replace(',', '.')



    FutureTable.Individuals_Long_Contract  = FutureTable.Individuals_Long_Contract.astype('int64')

    FutureTable.Individuals_Short_Contract = FutureTable.Individuals_Short_Contract.astype('int64')

    FutureTable.Legals_Long_Contract       = FutureTable.Legals_Long_Contract.astype('int64')

    FutureTable.Legals_Short_Contract      = FutureTable.Legals_Short_Contract.astype('int64')

    FutureTable.All_Sum_Contract           = FutureTable.All_Sum_Contract.astype('int64')



    FutureTable.Individuals_Long_Delta  = FutureTable.Individuals_Long_Delta.astype('int64')

    FutureTable.Individuals_Short_Delta = FutureTable.Individuals_Short_Delta.astype('int64')

    FutureTable.Legals_Long_Delta       = FutureTable.Legals_Long_Delta.astype('int64')

    FutureTable.Legals_Short_Delta      = FutureTable.Legals_Short_Delta.astype('int64')

    FutureTable.All_Sum_Delta           = FutureTable.All_Sum_Delta.astype('int64')



    FutureTable.Individuals_Long_Percent  = FutureTable.Individuals_Long_Percent.astype('float64')

    FutureTable.Individuals_Short_Percent = FutureTable.Individuals_Short_Percent.astype('float64')

    FutureTable.Legals_Long_Percent       = FutureTable.Legals_Long_Percent.astype('float64')

    FutureTable.Legals_Short_Percent      = FutureTable.Legals_Short_Percent.astype('float64')

    FutureTable.All_Sum_Percent           = FutureTable.All_Sum_Percent.astype('float64')



    FutureTable.Individuals_Long_Entities  = FutureTable.Individuals_Long_Entities.astype('int64')

    FutureTable.Individuals_Short_Entities = FutureTable.Individuals_Short_Entities.astype('int64')

    FutureTable.Legals_Long_Entities       = FutureTable.Legals_Long_Entities.astype('int64')

    FutureTable.Legals_Short_Entities      = FutureTable.Legals_Short_Entities.astype('int64')

    FutureTable.All_Sum_Entities           = FutureTable.All_Sum_Entities.astype('int64')

    

    FutureTable.TRADEDATE = pd.to_datetime(FutureTable.TRADEDATE, format='%Y-%m-%d')

    FutureTable = FutureTable[(FutureTable.TRADEDATE >= StartDate) & (FutureTable.TRADEDATE <= EndDate)]

    FutureTable = FutureTable.reset_index(drop=True)

    

    return FutureTable    



def addWeekend(X):

    X['TRADEDAY'] = 1   

    X['DELTADATE'] = 0



    D_Lag = X[['TRADEDATE']]

    D_Lag.loc[len(D_Lag.index)] = [np.NaN]

    D_Lag = D_Lag[1:]

    D_Lag.index = D_Lag.index - 1

    D_Lag.columns = ['TRADEDATE_2']

    X = X.join(D_Lag, how='left')



    X.DELTADATE = (X.TRADEDATE_2 - X.TRADEDATE).dt.days.astype('float64')

    X_Temp = X[X.DELTADATE > 1]



    for i in X_Temp.index:

        X_Part1 = X[X.TRADEDATE <= X_Temp.loc[i].TRADEDATE]

        X_Part2 = X[X.TRADEDATE >  X_Temp.loc[i].TRADEDATE]



        NewRowsNumb = X_Temp.DELTADATE[i] - 1



        for j in np.arange(NewRowsNumb) + 1:

            Row_Temp = X_Temp.loc[i]

            

            Row_Temp['TRADEDATE'] = Row_Temp['TRADEDATE'] + pd.DateOffset(j)

            Row_Temp['TRADEDAY'] = 0

            

            X_Part1 = X_Part1.append(Row_Temp)

            X_Part1 = X_Part1.reset_index(drop=True)



        X_Part2.index = X_Part2.index + NewRowsNumb

        X = X_Part1.append(X_Part2)



    X = X.drop(columns = {'DELTADATE', 'TRADEDATE_2'})

    X.index = X.index.astype('int')

    

    return X



def addFuture(X, FutureTable):

    X = X.merge(FutureTable, how='left', on='TRADEDATE')

    

    return X



def addLags(X, Lags):

    X_LagColumns = getLagColumns(X)

    X_Lag = X[X_LagColumns]



    for i in Lags:

        X_Lag.columns = X_LagColumns

        

        X_Lag.loc[-1] = np.tile(np.NaN, X_LagColumns.size)

        X_Lag.index = X_Lag.index + 1

        X_Lag = X_Lag.sort_index()

        X_Lag = X_Lag[:-1]



        X_Lag = X_Lag.add_suffix('_Lag' + str(i))

        X = X.join(X_Lag, how='left')

   

    return X



def addMonthAndSeason(X, MonthAndSeasonColumnNameIndex = ''):

    MonthColumnName  = 'MONTH'  + MonthAndSeasonColumnNameIndex

    SeasonColumnName = 'SEASON' + MonthAndSeasonColumnNameIndex

    

    X[MonthColumnName] = pd.to_datetime(X.TRADEDATE, format='%Y-%m-%d')

    X[MonthColumnName] = X[MonthColumnName].dt.month



    X[SeasonColumnName] = 0

    X.loc[(X[MonthColumnName] == 12)|(X[MonthColumnName] ==  1)|(X[MonthColumnName] ==  2), [SeasonColumnName]] = 1

    X.loc[(X[MonthColumnName] ==  3)|(X[MonthColumnName] ==  4)|(X[MonthColumnName] ==  5), [SeasonColumnName]] = 2

    X.loc[(X[MonthColumnName] ==  6)|(X[MonthColumnName] ==  7)|(X[MonthColumnName] ==  8), [SeasonColumnName]] = 3

    X.loc[(X[MonthColumnName] ==  9)|(X[MonthColumnName] == 10)|(X[MonthColumnName] == 11), [SeasonColumnName]] = 4

    

    X = X[list(X.columns[0:1]) + list(X.columns[-2:]) + list(X.columns[1:-2])] # перемещаем столбцы MONTH и SEASON в начало таблицы после TRADEDATE

    

    return X



def addTradingFeatures(X, ColumnNameClose = 'CLOSE', ColumnNameLow = 'LOW', ColumnNameHigh = 'HIGH', TradingFeaturesColumnNameIndex = ''):   

    I = [12, 20, 50, 100, 200, 250] # EMA 12 20 50 100 200 250



    for i in I: # MA, ЕМА, MACD, RSI, линии Боллинджера, Стохастик

        MA = pd.DataFrame(ta.MA(X[ColumnNameClose], i))

        MA.columns = ['MA' + str(i) + TradingFeaturesColumnNameIndex]

        MA.index = MA.index + (len(X.index)-len(MA.index))

        MA = round(MA, 2)

        X = X.join(MA, how='left')  

        

        EMA = pd.DataFrame(ta.EMA(X[ColumnNameClose], i))

        EMA.columns = ['EMA' + str(i) + TradingFeaturesColumnNameIndex]

        EMA.index = EMA.index + (len(X.index)-len(EMA.index))

        EMA = round(EMA, 2)

        X = X.join(EMA, how='left')



    RSI = pd.DataFrame(ta.RSI(X[ColumnNameClose]))

    RSI.columns = ['RSI' + TradingFeaturesColumnNameIndex]

    RSI = round(RSI, 2)

    X = X.join(RSI, how='left')

        

    MACD = pd.DataFrame(list(ta.MACD(X[ColumnNameClose]))).transpose()

    MACD.columns = ['MACD_FAST'   + TradingFeaturesColumnNameIndex,

                    'MACD_SLOW'   + TradingFeaturesColumnNameIndex,

                    'MACD_SIGNAL' + TradingFeaturesColumnNameIndex]

    MACD = round(MACD, 2)

    X = X.join(MACD, how='left')

    

    STOCH = pd.DataFrame(list(ta.STOCH(X[ColumnNameHigh], X[ColumnNameLow], X[ColumnNameClose], fastk_period=14))).transpose()

    STOCH.columns = ['STOCH_STOK' + TradingFeaturesColumnNameIndex,

                     'STOCH_STOD' + TradingFeaturesColumnNameIndex]

    STOCH = round(STOCH, 2)

    X = X.join(STOCH, how='left')

    

    BBANDS = pd.DataFrame(list(ta.BBANDS(X[ColumnNameClose], timeperiod=20, matype=ta.MA_Type.SMA))).transpose()

    BBANDS.columns = ['BBANDS_UPPER'  + TradingFeaturesColumnNameIndex,

                      'BBANDS_MIDDLE' + TradingFeaturesColumnNameIndex,

                      'BBANDS_LOWER'  + TradingFeaturesColumnNameIndex]

    BBANDS = round(BBANDS, 2)

    X = X.join(BBANDS, how='left')

    

    return X



def addMathFeatures(X, Columns = ['CLOSE', 'VOLUME'], MathFeaturesColumnNameIndex = ''): 

    dayWindows = [7,14,28,91,182,364]

    

    for column in Columns:

        Sum  = X[column].sum()

        Mean = X[column].mean()

        Min  = X[column].min()

        Max  = X[column].max()

        

        for window in dayWindows:

            X[column + '_Max_'    + str(window) + MathFeaturesColumnNameIndex] = X[column].rolling(window).max()

            X[column + '_Min_'    + str(window) + MathFeaturesColumnNameIndex] = X[column].rolling(window).min()

            X[column + '_Mean_'   + str(window) + MathFeaturesColumnNameIndex] = X[column].rolling(window).mean()

            X[column + '_Median_' + str(window) + MathFeaturesColumnNameIndex] = X[column].rolling(window).median()

            X[column + '_Sum_'    + str(window) + MathFeaturesColumnNameIndex] = X[column].rolling(window).sum()

            

            X[column + '_DayAfterMax_'     + str(window) + MathFeaturesColumnNameIndex] = window - 1 - X[column].rolling(window).apply(np.argmax)

            X[column + '_DayAfterMin_'     + str(window) + MathFeaturesColumnNameIndex] = window - 1 - X[column].rolling(window).apply(np.argmin)

            

            X[column + '_SumOnSumTotal_'   + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Sum_'  + str(window) + MathFeaturesColumnNameIndex] / Sum

            X[column + '_SumOnSumMean_'    + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Sum_'  + str(window) + MathFeaturesColumnNameIndex] / X[column + '_Sum_'  + str(window) + MathFeaturesColumnNameIndex].mean()

            

            X[column + '_MeanOnMeanTotal_' + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Mean_' + str(window) + MathFeaturesColumnNameIndex] / Mean          

            X[column + '_MeanOnSumTotal_'  + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Mean_' + str(window) + MathFeaturesColumnNameIndex] / Sum  

            X[column + '_MeanOnSum_'       + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Mean_' + str(window) + MathFeaturesColumnNameIndex] / X[column + '_Sum_'  + str(window) + MathFeaturesColumnNameIndex]

            

            X[column + '_MaxOnMin_'        + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Max_'  + str(window) + MathFeaturesColumnNameIndex] / X[column + '_Min_'  + str(window) + MathFeaturesColumnNameIndex]

            

            X[column + '_MaxOnMaxTotal_'   + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Max_'  + str(window) + MathFeaturesColumnNameIndex] / Max

            X[column + '_MaxOnMean_'       + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Max_'  + str(window) + MathFeaturesColumnNameIndex] / X[column + '_Mean_' + str(window) + MathFeaturesColumnNameIndex]

            

            X[column + '_MinOnMinTotal_'   + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Min_'  + str(window) + MathFeaturesColumnNameIndex] / Min

            X[column + '_MinOnMean_'       + str(window) + MathFeaturesColumnNameIndex] = X[column + '_Min_'  + str(window) + MathFeaturesColumnNameIndex] / X[column + '_Mean_' + str(window) + MathFeaturesColumnNameIndex]

            

            MeanSumTheseWindows = X[column + '_Sum_' + str(window) + MathFeaturesColumnNameIndex].mean()

            

            X_Temp = X[[column + '_Mean_'  + str(window) + MathFeaturesColumnNameIndex,     column + '_Sum_'  + str(window) + MathFeaturesColumnNameIndex]]

            X_Temp = X_Temp.reset_index(drop=False)

            X_Temp.columns = ['TEMP', 'MEAN', 'SUM']

            X_Temp = addLags(X_Temp, [2])

            

            X[column + '_DeltaPrevMeanAndThisMean_'           + str(window) + MathFeaturesColumnNameIndex] = (X_Temp['MEAN_Lag2'] - X_Temp['MEAN']) / X_Temp['MEAN']

            X[column + '_DeltaPrevSumAndThisSum_'             + str(window) + MathFeaturesColumnNameIndex] = (X_Temp ['SUM_Lag2'] - X_Temp ['SUM']) / X_Temp ['SUM']

            X[column + '_DeltaThisSumAndMeanSumTheseWindows_' + str(window) + MathFeaturesColumnNameIndex] = (X[column + '_Sum_'  + str(window) + MathFeaturesColumnNameIndex] - MeanSumTheseWindows) / MeanSumTheseWindows

            X[column + '_DeltaThisMeanAndMeanTotal_'          + str(window) + MathFeaturesColumnNameIndex] = (X[column + '_Mean_' + str(window) + MathFeaturesColumnNameIndex] - Mean) / Mean            

    

    return X



def prepareData(StockTable, FutureTable):

    # выбираем нужные столбцы и переименовываем их для процедур

    X = StockTable[StockTableColumnsForUse]

    

    # добавляем трейдерские фичи

    #X = addTradingFeatures(X)



    # добавляем математические фичи

    #X = addMathFeatures(X)



    # добавляем столбцы месяца и сезона, меняем порядок столбцов на более наглядный

    #X = addMonthAndSeason(X)



    # заполняем пропуски на месте выходных и праздничных дней

    #X = addWeekend(X) # в выходные и праздничные дни биржа не работает, поэтому показатели не меняются, включая месяц и сезон



    # добавляем фичи фьючерса

    #X = addFuture(X, FutureTable) # данные по фьючам имеют другую природу и только добавляют шум, ухудшая показатели



    # сохраняем список колонок, которые не должна видеть обучающая выборка

    global XColumnsWithoutLags

    XColumnsWithoutLags = X.columns



    # добавляем лаги - смещенные на день данные

    #Lags = np.arange(LagsNumb) + 2

    #X = addLags(X, Lags)

    

    # убираем строки с NaN, появившиеся при расчете фич с искользованием скользящего окна и при добавлении лагов

    #X = X.dropna()

    

    return X



def fitModel(X):

    global XColumnsWithLags

    XColumnsWithLags = X.columns



    #X_Train, X_Test  = train_test_split(X, test_size = 0) # 5% процентов от выборки отложим для теста модели

    

    X_Train = X

    X_Test = pd.DataFrame(columns=X.columns)

    

    print(len(X_Train))

    print(len(X_Test))

    

    X_train = X_Train[XColumnsWithLags]

    X_train = X_train.drop(columns=XColumnsWithoutLags)

    Y_train = X_Train[['CLOSE']]

    X_train = X_train.reset_index(drop=True)

    Y_train = Y_train.reset_index(drop=True)



    X_test = X_Train

    Y_test = Y_train

    

    t1 = datetime.today()

    print(t1)



    gbr = GradientBoostingRegressor(random_state=0)



    param_grid = {

        'n_estimators':      [7000],

        'learning_rate':     [0.01],

        'max_depth':         [len(X_train.columns) + 1],

        'max_features':      ['sqrt'],

        'min_samples_leaf':  [15],

        'min_samples_split': [10],

        'loss':              ['huber'],

        'random_state':      [42]

    }

    

    model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=-1, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

    

    with parallel_backend('multiprocessing'):

        model.fit(X_train, Y_train.values.ravel())

    

    print(model.scorer_)

    

    print('Best CV Score:')

    print(model.best_score_)

    

    print('CV Results:')

    print(model.cv_results_)

    

    t2 = datetime.today()

    print(t2)

    

    print(t2-t1)

    

    return model, X_test, Y_test



def predictAndPlot(X, model, PlotLabel, FileName):

    XBuf = X[XColumnsWithLags]

    XBuf = XBuf.reset_index(drop=True)



    X_pred = XBuf[XColumnsWithLags]

    X_pred = X_pred.drop(columns=XColumnsWithoutLags)



    LastRow = XBuf[XBuf.index == len(XBuf.index) - 1]

    LastRow.index = LastRow.index + 1

    LastRow = LastRow.drop(columns='TRADEDATE')

    LastRow = LastRow.drop(columns=(XColumnsWithoutLags + '_Lag' + str(LagsNumb + 1)).drop(labels='TRADEDATE_Lag' + str(LagsNumb + 1)))

    LastRow.columns = X_pred.columns



    X_pred = X_pred.append(LastRow)

    Y_pred = pd.DataFrame(model.predict(X_pred), columns={'PREDICT'})



    LastRowFake = XBuf[XBuf.index == len(XBuf.index) - 1]

    LastRowFake.index = LastRowFake.index + 1

    LastRowFake['TRADEDATE'] = LastRowFake['TRADEDATE'] + timedelta(days=1)



    XBuf = XBuf.append(LastRowFake)

    Y_pred = Y_pred.join(XBuf, how='left')

    

    plotPredictionResult(X, Y_pred, PlotLabel, FileName)



def plotPredictionResult(TrainTable, PredictTable, title, filename):

    trace_close = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.CLOSE,

        name = "Close",

        line = dict(color = '#66CD00'),

        opacity = 0.8)  

    trace_predict = go.Scatter(

        x=PredictTable.TRADEDATE,

        y=PredictTable.PREDICT,

        name = "Predict",

        line = dict(color = 'red'),

        opacity = 0.8)

    

    trace_EMA12 = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.EMA12,

        name = "EMA 12",

        line = dict(color = 'red'),

        opacity = 0.1)

    trace_EMA50 = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.EMA50,

        name = "EMA 50",

        line = dict(color = 'blue'),

        opacity = 0.1)

    trace_EMA200 = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.EMA200,

        name = "EMA 200",

        line = dict(color = 'black'),

        opacity = 0.1)

    

    trace_MA20 = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.BBANDS_MIDDLE,

        name = "MA 20",

        line = dict(color = '#66CD00'),

        opacity = 0.2)

    trace_PlusSTDx2 = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.BBANDS_UPPER,

        name = "MA + 2STD",

        line = dict(color = '#66CD00'),

        opacity = 0.2)

    trace_MinusSTDx2 = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.BBANDS_LOWER,

        name = "MA - 2STD",

        line = dict(color = '#66CD00'),

        opacity = 0.2)

    

    trace_RSI = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.RSI,

        name = "RSI",

        line = dict(color = 'red'),

        opacity = 0.1)

    

    trace_MACD_FAST = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.MACD_FAST,

        name = "MACD_FAST",

        line = dict(color = 'green'),

        opacity = 0.1)

    trace_MACD_SLOW = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.MACD_SLOW,

        name = "MACD_SLOW",

        line = dict(color = 'green'),

        opacity = 0.1)

    trace_MACD_SIGNAL = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.MACD_SIGNAL,

        name = "MACD_SIGNAL",

        line = dict(color = 'green'),

        opacity = 0.1)

    

    trace_STOCH_STOK = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.STOCH_STOK,

        name = "STOCH_STOK",

        line = dict(color = 'blue'),

        opacity = 0.1)

    trace_STOCH_STOD = go.Scatter(

        x=TrainTable.TRADEDATE,

        y=TrainTable.STOCH_STOD,

        name = "STOCH_STOD",

        line = dict(color = 'blue'),

        opacity = 0.1)

    

    data = [trace_predict,

            trace_close,

            trace_EMA12,

            trace_EMA50,

            trace_EMA200,

            trace_MA20,

            trace_PlusSTDx2,

            trace_MinusSTDx2,

            #trace_RSI,

            #trace_MACD_FAST,

            #trace_MACD_SLOW,

            #trace_MACD_SIGNAL,

            #trace_STOCH_STOK,

            #trace_STOCH_STOD

           ]

            

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

    

    print(title + ' last '    + str(TrainTable.iloc  [-1].TRADEDATE.date()) + ': ' + str(TrainTable.iloc  [-1].CLOSE))

    print(title + ' predict ' + str(PredictTable.iloc[-1].TRADEDATE)        + ': ' + str(PredictTable.iloc[-1].PREDICT))

    

def threadFunc(ThreadInd, StockName, FutureName):

    StartDate = '2010-01-01'

    EndDate = '2019-12-04' #datetime.today().date()    

    

    FileName = StockName + '_predict.html'

    

    StockTable = loadStockTable(StockName, StartDate, EndDate)

    FutureTable = loadFutureTable(FutureName, StartDate, EndDate)

    

    MaxFutureDate = FutureTable.TRADEDATE.max()

    MinFutureDate = FutureTable.TRADEDATE.min()



    StockTable = StockTable[(StockTable.TRADEDATE >= MinFutureDate) & (StockTable.TRADEDATE <= MaxFutureDate)]

    

    Data = prepareData(StockTable, FutureTable)

    Data = pd.read_csv("../input/moex-gazp-dataset/dataset.csv").drop(columns=['Unnamed: 0'])

    

    Model, X_test, Y_test = fitModel(Data)

    

    #Y_pred = pd.DataFrame(Model.predict(X_test), columns={'PREDICT'})

    #Y_pred['CLOSE'] = Y_test

    #Y_pred['CLOSE_Lag2'] = X_test['CLOSE_Lag2']

    #Y_pred['DELTA'] = Y_pred['CLOSE'] - Y_pred['PREDICT']

    

    #Y_pred['OK'] = 0

    #Y_pred.loc[(Y_pred.CLOSE > Y_pred.CLOSE_Lag2) & (Y_pred.PREDICT > Y_pred.CLOSE_Lag2), 'OK'] = 1

    #Y_pred.loc[(Y_pred.CLOSE < Y_pred.CLOSE_Lag2) & (Y_pred.PREDICT < Y_pred.CLOSE_Lag2), 'OK'] = 1

    #Y_pred.loc[(Y_pred.CLOSE == Y_pred.CLOSE_Lag2), 'OK'] = -1

    

    #print(Y_pred.OK.value_counts())    

    #print(Y_pred)

    

    #print(len(Data.columns))

    

    predictAndPlot(Data, Model, StockName, FileName)

    #plot_features(Model, (10, 500))

    

    # код по расчету уровня значимости



def main(StockNames, FutureNames):

    StockNumb = len(StockNames)

    

    for StockInd in range(StockNumb):

        print('Start: ' + StockNames[StockInd] + ' ' + str(datetime.today()))

        threadFunc(StockInd, StockNames[StockInd], FutureNames[StockInd])

        print('Finish: ' + StockNames[StockInd] + ' ' + str(datetime.today()))   
# настройки создания фичей              - в глобальных переменных

# настройки обучаемой модели            - в переменных процедуры потока threadFunc

# настройки списка акций и их обработки - в переменных процедуры main



StockNames = [

    'GAZP',  # Газпром      ао

    #'LKOH',  # Лукойл       ао

    #'SBER',  # Сбербанк     ао

    #'ROSN',  # Роснефть     ао

    #'BANEP', # Башнефть     ап

    #'TATNP', # Татнефть     ап

    #'SIBN',  # Газпромнефть ао

    #'CHMF',  # Северсталь   ао

    ]



FutureNames = [

    'GAZR_F', # Фьючерсный контракт на обыкновенные акции ПАО "Газпром"

    ]



main(StockNames, FutureNames)