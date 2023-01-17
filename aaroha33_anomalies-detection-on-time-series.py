import datetime

sttime = datetime.datetime.now()

import pandas as pd

import numpy as np

import plotly.graph_objs as go

import math

from sklearn.metrics import mean_squared_error

!pip install pmdarima

from pmdarima.arima import auto_arima

!pip install sesd

import sesd
dataset_train = pd.read_excel('../input/anomalies-detection/Anomalies detection.xlsx')
df= dataset_train[['Port','Temp', 'RecorDate']]

df_filtered = df.loc[lambda x: x['Port'] == 'Port 0']

KPI_Series = df_filtered[['RecorDate','Temp']]
KPI_Series .columns = ['load_date','actuals']

KPI_Series = KPI_Series.sort_values(by="load_date")

KPI_Series = KPI_Series.reset_index(drop=True)

actual_vals = KPI_Series.actuals.values

actual_vals= actual_vals[~np.isnan(actual_vals)]

train, test = actual_vals[0:-70], actual_vals[-70:]

train_log, test_log = np.log10(train), np.log10(test)

train_log= train_log[~np.isnan(train_log)]

outliers_indices = sesd.seasonal_esd(actual_vals, hybrid=False, max_anomalies=10, alpha=0.05)
KPI_Series.load_date = pd.to_datetime(KPI_Series.load_date, format='%Y%m%d')

KPI_Series = KPI_Series.sort_values(by="load_date")

KPI_Series = KPI_Series.reset_index(drop=True)
actual_vals = KPI_Series.actuals.values

actual_log = np.log10(actual_vals)
train, test = actual_vals[0:-70], actual_vals[-70:]

train_log, test_log = np.log10(train), np.log10(test)

train_log= train_log[~np.isnan(train_log)]
stepwise_model = auto_arima(train_log, start_p=1, start_q=1,

                           max_p=3, max_q=3, m=1,

                           start_P=0, seasonal=True,

                           d=1, D=1, trace=True,

                           error_action='ignore',  

                           suppress_warnings=True, 

                           stepwise=True)



output = stepwise_model.predict(n_periods=100)



history = [x for x in train_log]

predictions = list()

predict_log=list()

for t in range(len(test_log)):

    print(t,len(test_log))

    #model = sm.tsa.SARIMAX(history, order=my_order, seasonal_order=my_seasonal_order,enforce_stationarity=False,enforce_invertibility=False)

    stepwise_model.fit(history)

    output = stepwise_model.predict(n_periods=1)

    predict_log.append(output[0])

    yhat = 10**output[0]

    predictions.append(yhat)

    obs = test_log[t]

    history.append(obs)
print('predicted=%f, expected=%f' % (output[0], obs))

error = math.sqrt(mean_squared_error(test_log, predict_log))

print('Test rmse: %.3f' % error)
predicted_df=pd.DataFrame()

predicted_df['load_date']=KPI_Series['load_date'][-70:]

predicted_df['actuals']=test

predicted_df['predicted']=predictions

predicted_df.reset_index(inplace=True)

del predicted_df['index']



#Predictin for next 5 Days 

predictions_5day = list()

predict_log_5day=list()

for t in range(120):

    print(t,'/',120)

    stepwise_model.fit(history)

    output = stepwise_model.predict(n_periods=1)

    predict_log_5day.append(output[0])

    yhat = 10**output[0]

    predictions_5day.append(yhat)

    history.append(obs)

    

# Generating output data

predicted_df_5day=pd.DataFrame()

 #edtime = 2019-10-31 15:00:00 
def detect_classify_anomalies(df,window):

    df.replace([np.inf, -np.inf], np.NaN, inplace=True)

    df.fillna(0,inplace=True)

    df['error']=df['actuals']-df['predicted']

    df['percentage_change'] = ((df['actuals'] - df['predicted']) / df['actuals']) * 100

    df['meanval'] = df['error'].rolling(window=window).mean()

    df['deviation'] = df['error'].rolling(window=window).std()

    df['-3s'] = df['meanval'] - (2 * df['deviation'])

    df['3s'] = df['meanval'] + (2 * df['deviation'])

    df['-2s'] = df['meanval'] - (1.75 * df['deviation'])

    df['2s'] = df['meanval'] + (1.75 * df['deviation'])

    df['-1s'] = df['meanval'] - (1.5 * df['deviation'])

    df['1s'] = df['meanval'] + (1.5 * df['deviation'])

    cut_list = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]

    cut_values = cut_list.values

    cut_sort = np.sort(cut_values)

    df['impact'] = [(lambda x: np.where(cut_sort == df['error'][x])[1][0])(x) for x in

                               range(len(df['error']))]

    severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}

    region = {0: "NEGATIVE", 1: "NEGATIVE", 2: "NEGATIVE", 3: "NEGATIVE", 4: "POSITIVE", 5: "POSITIVE", 6: "POSITIVE",

              7: "POSITIVE"}

    df['color'] =  df['impact'].map(severity)

    df['region'] = df['impact'].map(region)

    df['anomaly_points'] = np.where(df['color'] == 3, df['error'], np.nan)

    df = df.sort_values(by='load_date', ascending=False)

    df.load_date = pd.to_datetime(df['load_date'].astype(str), format="%Y-%m-%d")

    return df



classify_df=detect_classify_anomalies(predicted_df,7)

classify_df.reset_index(inplace=True)

del classify_df['index']



'''

loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.score(X_test, Y_test)'''