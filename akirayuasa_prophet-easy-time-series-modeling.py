#!pip install fbprophet
import numpy as np

import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

from fbprophet import Prophet



covid_data = pd.read_csv("../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv")

df = pd.read_csv("../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-01-02_To2020-06-08.csv")



#入力データを絞り、Prophetのフォーマットに変換する

df = df[-1000:] # train期間を絞る

df['ds'] = pd.to_datetime(df['Date']).dt.date #date型に変換

#df['y']  = np.log(df['Price']) #データの変化が大きい場合は、対数変換をする。

df['y']  = df['Price']

df = df[['ds', 'y']] #元データは土日が含まれていないことに注意



#trainとtestを分離する

test_y = df[-30:] #test

df = df[:-30] # train

df['y'].plot()
#まずはデフォルトパラメータでモデルを作成する

m = Prophet()

m.fit(df)
#予測をするためのデータフレームを作る

#periodに期間を入れる。この場合50日分の予測値が出力される(土日を含めて予測されることに注意)

future = m.make_future_dataframe(periods=50, freq = 'd')

print(future)



#予測をする

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(20)

#予測結果をグラフ表示

m.plot(forecast);
#全体は長過ぎるので絞って表示するようにする．

#m.plot(forecast);

#plt.xlim(future.ds.iloc[-100], future.ds.iloc[-1])
# 予測のインタラクティブな図は、plotlyで作成できる。

"""

from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast) 

py.iplot(fig)

"""
pred = forecast[['ds', 'yhat']][-40:]

pred['ds'] = pd.to_datetime(pred['ds']).dt.date #date型に変換



result = pd.merge(test_y, pred, how="inner" ,on="ds")



result
#RMSE

from sklearn.metrics import mean_squared_error



rmse = np.sqrt(mean_squared_error(result['y'] , result['yhat'] ))

print('Test RMSE: %.3f' % rmse)

#パラメータを指定し、モデルを作成

m = Prophet(growth='logistic', #ロジスティックモデル

            n_changepoints=40, #変更点の数=40

            changepoint_range=1, #データの全範囲で変更点検知する。

            changepoint_prior_scale=0.5, #結構オーバーフィッティング気味にする

            weekly_seasonality=False,

            yearly_seasonality=True)

df['cap'] = 130

df['floor'] = 0

m.fit(df)



#予測をするためのデータフレームを作る

#periodに期間を入れる。この場合365日分の予測値が出力される

#logisticのときは、予測の際も未来のcapを指定する

future = m.make_future_dataframe(periods=730, freq = 'd')

future['cap'] = 130

future['floor'] = 0



#予測をする

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()



#予測結果をグラフ表示

m.plot(forecast);
#全体は長過ぎるので直近のみを表示するようにする．

#m.plot(forecast);

#plt.xlim(future.ds.iloc[-100], future.ds.iloc[-1])
#changepointの可視化

from fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
#自動検出された変更点と、変化量をチェック

print(m.changepoints)

m.params['delta']
pred = forecast[['ds', 'yhat']][-1000:]

pred['ds'] = pd.to_datetime(pred['ds']).dt.date #date型に変換

# inner joinで評価期間のデータを抽出

result = pd.merge(test_y, pred, how="inner" ,on="ds")



result


#RMSE

from sklearn.metrics import mean_squared_error



rmse = np.sqrt(mean_squared_error(result['y'] , result['yhat'] ))

print('Test RMSE: %.3f' % rmse)
m.plot_components(forecast);
from fbprophet import diagnostics



#initialはtrainの期間。そこからhorizon期間分を予測する。これをperiodの期間スライドし、何度も実行する。

#cv = diagnostics.cross_validation(m, horizon='30 days')

cv = diagnostics.cross_validation(m, initial = '0 days', period = '100 days', horizon = '30 days')

cv.tail()
from fbprophet.diagnostics import performance_metrics

df_p = performance_metrics(cv)

df_p
from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(cv, metric='rmse')
#MAPE

def cal_mape(df):

    return((df['yhat'] - df['y']).div(df['y']).abs().sum()*(1/len(df)))

 

print('Test MAPE: %.3f' % cal_mape(cv))

#RMSE

from sklearn.metrics import mean_squared_error



rmse = np.sqrt(mean_squared_error(cv['y'] , cv['yhat'] ))





print('Test RMSE: %.3f' % rmse)
