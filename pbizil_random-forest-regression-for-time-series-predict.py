import os 

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller
arqs = os.listdir("../input/stock-time-series-20050101-to-20171231/")

for a in ['all_stocks_2006-01-01_to_2018-01-01.csv', 

          'all_stocks_2017-01-01_to_2018-01-01.csv']:

  arqs.remove(a)
data=pd.DataFrame(pd.read_csv("../input/stock-time-series-20050101-to-20171231/"+arqs[0])['Date'])

for a in arqs:

  name=a.replace('_2006-01-01_to_2018-01-01.csv','')

  arq = pd.DataFrame(pd.read_csv("../input/stock-time-series-20050101-to-20171231/"+a)["Open"])

  arq.rename(columns={'Open': name}, inplace=True)

  data=pd.concat([data, arq], axis=1, join='outer')
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,1], name = data.iloc[:,1].name, line = dict(color = '#17BECF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,2], name = data.iloc[:,2].name, line = dict(color = '#CF1717'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,3], name = data.iloc[:,3].name, line = dict(color = '#AACF17'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,4], name = data.iloc[:,4].name, line = dict(color = '#17CF29'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,5], name = data.iloc[:,5].name, line = dict(color = '#1742CF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,6], name = data.iloc[:,6].name, line = dict(color = '#B017CF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,7], name = data.iloc[:,7].name, line = dict(color = '#CF1773'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,8], name = data.iloc[:,8].name, line = dict(color = '#17B0CF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,9], name = data.iloc[:,9].name, line = dict(color = '#CFA417'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,10], name = data.iloc[:,10].name, line = dict(color = '#CF5E17'), opacity = 0.8))

fig.update_layout(title_text=data.iloc[:,1].name+', '+data.iloc[:,2].name+', '+data.iloc[:,3].name+', '+data.iloc[:,4].name+', '+data.iloc[:,5].name+', '+

                 data.iloc[:,6].name+', '+data.iloc[:,7].name+', '+data.iloc[:,8].name+', '+data.iloc[:,9].name+' and '+data.iloc[:,10].name)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,11], name = data.iloc[:,11].name, line = dict(color = '#17BECF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,12], name = data.iloc[:,12].name, line = dict(color = '#CF1717'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,13], name = data.iloc[:,13].name, line = dict(color = '#AACF17'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,14], name = data.iloc[:,14].name, line = dict(color = '#17CF29'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,15], name = data.iloc[:,15].name, line = dict(color = '#1742CF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,16], name = data.iloc[:,16].name, line = dict(color = '#B017CF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,17], name = data.iloc[:,17].name, line = dict(color = '#CF1773'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,18], name = data.iloc[:,18].name, line = dict(color = '#17B0CF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,19], name = data.iloc[:,19].name, line = dict(color = '#CFA417'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,20], name = data.iloc[:,20].name, line = dict(color = '#CF5E17'), opacity = 0.8))

fig.update_layout(title_text=data.iloc[:,11].name+', '+data.iloc[:,12].name+', '+data.iloc[:,13].name+', '+data.iloc[:,14].name+', '+data.iloc[:,15].name+', '+

                 data.iloc[:,16].name+', '+data.iloc[:,17].name+', '+data.iloc[:,18].name+', '+data.iloc[:,19].name+' and '+data.iloc[:,20].name)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,21], name = data.iloc[:,21].name, line = dict(color = '#17BECF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,22], name = data.iloc[:,22].name, line = dict(color = '#CF1717'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,23], name = data.iloc[:,23].name, line = dict(color = '#AACF17'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,24], name = data.iloc[:,24].name, line = dict(color = '#17CF29'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,25], name = data.iloc[:,25].name, line = dict(color = '#1742CF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,26], name = data.iloc[:,26].name, line = dict(color = '#B017CF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,27], name = data.iloc[:,27].name, line = dict(color = '#CF1773'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,28], name = data.iloc[:,28].name, line = dict(color = '#17B0CF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,29], name = data.iloc[:,29].name, line = dict(color = '#CFA417'), opacity = 0.8))

fig.add_trace(go.Scatter(x=data.Date, y=data.iloc[:,30], name = data.iloc[:,30].name, line = dict(color = '#CF5E17'), opacity = 0.8))

fig.update_layout(title_text=data.iloc[:,21].name+', '+data.iloc[:,22].name+', '+data.iloc[:,23].name+', '+data.iloc[:,24].name+', '+data.iloc[:,25].name+', '+

                 data.iloc[:,26].name+', '+data.iloc[:,27].name+', '+data.iloc[:,28].name+', '+data.iloc[:,29].name+' and '+data.iloc[:,30].name)

fig.show()
variations=((data.select_dtypes(float)/data.select_dtypes(float).shift(1))-1)

variations=variations.dropna()
f = plt.figure(figsize=(15, 13))

plt.matshow(variations.corr(), fignum=f.number)

plt.xticks(range(variations.shape[1]), variations.columns, fontsize=14, rotation=45)

plt.yticks(range(variations.shape[1]), variations.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14);
data=data.set_index("Date")

data=data.dropna()

columns = data.columns
data=data.dropna()

index=data.index

columns = data.columns

df_stats = pd.DataFrame(columns=['P-value'], index=columns)

for col in columns:

  dftest = adfuller(data[col], autolag='AIC')

  df_stats.loc[col]=dftest[1]
import plotly.graph_objects as go

fig = go.Figure(data=[go.Bar(

            x=df_stats.index, y=df_stats['P-value'],

            text=df_stats['P-value'],

            textposition='auto',)])



fig.show()
data_diff = pd.DataFrame(columns=columns)

data_diff['Date']=data.index

for col in columns:

  data_diff[col] = pd.DataFrame(np.diff(data[col]))



data_diff=data_diff.dropna()

for col in columns:

  dftest = adfuller(data_diff[col], autolag='AIC')

  df_stats.loc[col]=dftest[1]
import plotly.graph_objects as go

fig = go.Figure(data=[go.Bar(

            x=df_stats.index, y=df_stats['P-value'],

            text=df_stats['P-value'],

            textposition='auto',)])



fig.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=data_diff.Date, y=data_diff.iloc[:,1], name = data_diff.iloc[:,1].name, line = dict(color = '#17BECF'), opacity = 0.8))
!pip install pmdarima
from pmdarima.arima import auto_arima

stepwise_model = auto_arima(data_diff["AAPL"], start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True,

                           d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
data_diff=data_diff.set_index('Date')

train = data_diff["AAPL"].loc[data_diff.index < '2016-12-31']

test = data_diff["AAPL"].loc['2016-12-31':'2017-01-31']

stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=20)

future_forecast = pd.DataFrame(future_forecast, index = test.index, columns=["AAPL"])
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=test.index, y=test.values, name = "AAPL - Original Serie", line = dict(color = '#17BECF'), opacity = 0.8))

fig.add_trace(go.Scatter(x=future_forecast.index, y=future_forecast["AAPL"], name = "AAPL - Forecasting Serie", line = dict(color = '#CF1717'), opacity = 0.8))

fig.show()
lags = pd.DataFrame()

for i in range(10,0,-1):

    lags['t-'+str(i)] = data_diff["AAPL"].shift(i)

    lags['t'] = data_diff['AAPL'].values

lags = lags[13:]



from sklearn.ensemble import RandomForestRegressor

array = lags.values

X = array[:,0:-1]

y = array[:,-1]

model = RandomForestRegressor(n_estimators=500, random_state=1)

model.fit(X, y)
import plotly.graph_objects as go

names = lags.columns

fig = go.Figure(data=[go.Bar(

            x=lags.columns, y=model.feature_importances_,

            text=model.feature_importances_,

            textposition='auto',)])



fig.show()
from sklearn.feature_selection import RFE

rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 4)

fit = rfe.fit(X, y)

names = lags.columns

columns=[]

for i in range(len(fit.support_)):

    if fit.support_[i]:

        columns.append(names[i])



print("Columns with predictive power:", columns )
from statsmodels.tsa.seasonal import seasonal_decompose

import plotly.graph_objects as go

result = seasonal_decompose(data, model='additive', freq=1)

fig = go.Figure()

fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal.values, mode='lines', name='Seasonal - AAPL'))
columns=data.columns

data_mean=pd.DataFrame(columns=columns)

for col in columns:

    data_mean[col] = data[col].rolling(window = 80).mean()

data_mean = data_mean.dropna()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=data.index, y=data['AAPL'], mode='lines', name='AAPL'))

fig.add_trace(go.Scatter(x=data.index, y=data_mean['AAPL'], mode='lines', name='AAPL - Rolling Mean'))
#pre-processing data

df_forecasting=pd.DataFrame(data["AAPL"])

df_forecasting["AAPL_diff"] = df_forecasting["AAPL"].diff()

for i in range(4,0,-1):

    df_forecasting['t-'+str(i)] = df_forecasting["AAPL"].shift(i)

df_forecasting=df_forecasting.dropna()

df_forecasting["AAPL_rolling"] = df_forecasting["AAPL"].rolling(window = 80).mean()

df_forecasting= df_forecasting.dropna()
#model

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

x=df_forecasting.iloc[:,1:]

y=df_forecasting.iloc[:,0]

x_train, x_valid = x.loc[x.index < '2017-10-01'], x.loc[x.index >= '2017-10-01']

y_train, y_valid = y.loc[y.index < '2017-10-01'], y.loc[y.index >= '2017-10-01']

mdl = rf=RandomForestRegressor(n_estimators=100)

mdl.fit(x_train, y_train)

pred=mdl.predict(x_valid)

mean_absolute_error(y_valid, pred)

pred=pd.Series(pred, index=y_valid.index)
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=y_valid.index, y=y_valid.values, mode='lines', name='AAPL'))

fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode='lines', name='AAPL - Forecasting'))
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=y_train.loc[y_train.index > '2017-07-01'].index, y=y_train.loc[y_train.index > '2017-07-01'].values, mode='lines', name='AAPL before'))

fig.add_trace(go.Scatter(x=y_valid.index, y=y_valid.values, mode='lines', name='AAPL observed'))

fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode='lines', name='AAPL - Forecasting'))
from sklearn.metrics import mean_squared_log_error

var_menor_erro = None

valor_menor_erro = 1000.



for var in x_train.columns:

    mdl = RandomForestRegressor(n_jobs=-1, random_state=0, n_estimators=500)

    mdl.fit(x_train[[var]], y_train)

    p = mdl.predict(x_valid[[var]])

    erro = np.sqrt(mean_squared_log_error(y_valid, p)) * 100

    print("Variável: {} - Erro: {:.4f}\n".format(var, erro))

    

    if erro < valor_menor_erro:

        var_menor_erro = var

        valor_menor_erro = erro

        

print("Var: {} - Error: {:.4f}\n".format(var_menor_erro, valor_menor_erro))
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

score=[]

mse=[]

predict=pd.DataFrame()

for c in data.columns:

  #pre-processing data

  df_forecasting=pd.DataFrame(data[c])

  df_forecasting["var_diff"] = df_forecasting[c].diff()

  for i in range(4,0,-1):

      df_forecasting['t-'+str(i)] = df_forecasting[c].shift(i)

  df_forecasting=df_forecasting.dropna()

  df_forecasting["var_rolling"] = df_forecasting[c].rolling(window = 80).mean()

  df_forecasting= df_forecasting.dropna()

  #modeling 

  x=df_forecasting.iloc[:,1:]

  y=df_forecasting.iloc[:,0]

  x_train, x_valid = x.loc[x.index < '2017-10-01'], x.loc[x.index >= '2017-10-01']

  y_train, y_valid = y.loc[y.index < '2017-10-01'], y.loc[y.index >= '2017-10-01']

  mdl = RandomForestRegressor(n_estimators=100)

  mdl.fit(x_train, y_train)

  pred=mdl.predict(x_valid)

  predict[c+"_valid"]=y_valid.values

  predict[c+"_predict"]=pred

  m=mean_absolute_error(y_valid, pred)

  s=mdl.score(x_valid, y_valid)

  score.append([c, s])

  mse.append([c, m])



predict=predict.set_index(y_valid.index)

pred=pd.Series(pred, index=y_valid.index)

score=pd.DataFrame(score, columns=["Asset", "Score"])

score=score.set_index("Asset")
import plotly.graph_objects as go

fig = go.Figure(data=[go.Bar(

            x=score.index, y=score["Score"],

            text=score["Score"],

            textposition='auto',)])



fig.show()
print("Max is:", score[score["Score"]==score.max()[0]].index)

print("Min is:", score[score["Score"]==score.min()[0]].index)
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=data["INTC"].loc['2017-07-01':'2017-10-01'].index, y=data["INTC"].loc['2017-07-01':'2017-10-01'].values, mode='lines', name='INTC observed'))

fig.add_trace(go.Scatter(x=predict["INTC_valid"].index, y=predict["INTC_valid"].values, mode='lines', name='INTC observed'))

fig.add_trace(go.Scatter(x=predict["INTC_predict"].index, y=predict["INTC_predict"].values, mode='lines', name='INTC - Forecasting'))
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=data["VZ"].loc['2017-07-01':'2017-10-01'].index, y=data["VZ"].loc['2017-07-01':'2017-10-01'].values, mode='lines', name='VZ observed'))

fig.add_trace(go.Scatter(x=predict["VZ_valid"].index, y=predict["VZ_valid"].values, mode='lines', name='VZ observed'))

fig.add_trace(go.Scatter(x=predict["VZ_predict"].index, y=predict["VZ_predict"].values, mode='lines', name='VZ - Forecasting'))