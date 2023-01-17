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
from sklearn.model_selection import train_test_split
import plotly.express as px
path = '/kaggle/input/causes-of-death-in-50-69-year-olds.csv'

dataframe = pd.read_csv(path)
dataframe.head()
df=pd.concat( [dataframe[dataframe["Entity"]=="Mauritania"]], ignore_index=True)
df.head()
#df=df.transpose()
#df.tail(7)
df.shape
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
sns.pairplot(data=df)
df['sum']=df.sum(axis = 1, skipna = True)
df.tail(10)
data=df.sum(axis = 0, skipna = True)
data = pd.DataFrame(data)
data= data.iloc[1:]
data['maladie']=data.index
#data=data.transpose()
data.rename(columns={0:'cases'}, 
                 inplace=True)

data = data.reset_index(drop=True)
data= data.iloc[2:]
data
data= data[:-1]
data.shape
import plotly.express as px
#df = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
fig = px.pie(data, values='cases', names='maladie', title='les décés des maladies et épidémies dans la Mauritanie age = 50-69 ans')
fig.show()
fig = px.sunburst(data.sort_values(by='cases', ascending=False).reset_index(drop=True), path=["maladie"], values="cases", title='les décés des maladies et épidémies dans la Mauritanie age = 50-69 ans', color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()
temp = dataframe.groupby(['Year','Entity'])['CardiovascularDiseases'].sum().reset_index().sort_values('CardiovascularDiseases', ascending=False)
px.line(temp, x="Year", y="CardiovascularDiseases", color='Entity', title='CONFIRMED cases spread - Globally', height=400)
df_en=dataframe[dataframe["Entity"]=="Mauritania"]
temp = df_en.groupby(['Year','Entity'])['CardiovascularDiseases'].sum().reset_index().sort_values('CardiovascularDiseases', ascending=False)
px.line(temp, x="Year", y="CardiovascularDiseases", color='Entity', title='CONFIRMED cases spread - Globally', height=400)
fig = px.bar(data.sort_values('cases', ascending=False).sort_values('cases', ascending=True), 
             x="cases", y="maladie", 
             title='Total Active Cases', 
             text='cases', 
             orientation='h', 
             width=1000, height=700, range_x = [0, max(data['cases'])])
fig.update_traces(marker_color='red', opacity=0.8, textposition='inside')
fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')
fig.show()
df_en=dataframe[dataframe["Entity"]=="Mauritania"]
plt.plot(df_en["Year"]
         ,df_en["CardiovascularDiseases"]
        , color = 'blue'
        , label = 'Cardio vasculaires'
        , marker = 'o')

# defyning titles, labels and ticks parameters
plt.title('Les décées de maladies Cardio vasculaires',size=15)
plt.ylabel('Cases',size=20)
plt.xlabel('Updates',size=12)
plt.xticks(rotation=90,size=15)
plt.yticks(size=15)

# defyning legend parameters
plt.legend(loc = "upper left"
           , frameon = True
           , fontsize = 10
           , ncol = 1
           , fancybox = True
           , framealpha = 0.95
           , shadow = True
           , borderpad = 1);


plt.figure(figsize=(18,9))
plt.plot(df_en['Year'], df_en["CardiovascularDiseases"], linestyle="-")
plt.xlabel=('Dates')
plt.ylabel=('Total Production')
plt.show();
# prep data 

time_series_data = df_en[['Year', 'CardiovascularDiseases']].groupby('Year', as_index = False).sum()
time_series_data.columns = ['ds', 'y']
time_series_data.ds = pd.to_datetime(time_series_data.ds, format='%Y')
time_series_data.ds = pd.to_datetime(time_series_data.ds)

train_range = np.random.rand(len(time_series_data)) < 0.8
train_ts = time_series_data[train_range]
test_ts = time_series_data[~train_range]
test_ts = test_ts.set_index('ds')
from fbprophet import Prophet
m = Prophet()
m.fit(train_ts)
future = m.make_future_dataframe(periods=12,freq='Y')
prophet_pred = m.predict(future)
#prophet_pred = pd.DataFrame({"Date" : prophet_pred[-12:]['ds'], "Pred" : prophet_pred[-12:]["yhat"]})
prophet_pred = prophet_pred.set_index("ds")
prophet_pred['ds']=prophet_pred.index
# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = m.plot(prophet_pred, ax=ax)
# Plot the components of the model
fig = m.plot_components(prophet_pred)
import plotly.graph_objects as go

test_fig = go.Figure() 
test_fig.add_trace(go.Scatter(
                x= test_ts.index,
                y= test_ts.y,
                name = "Actual Cases",
                line_color= "deepskyblue",
                mode = 'lines',
                opacity= 0.8))
test_fig.add_trace(go.Scatter(
                x= prophet_pred.index,
                y= prophet_pred.yhat,
                name= "Prediction",
                mode = 'lines',
                line_color = 'red',
                opacity= 0.8))
test_fig.add_trace(go.Scatter(
                x= prophet_pred.index,
                y= prophet_pred.yhat_lower,
                name= "Prediction Lower Bound",
                mode = 'lines',
                line = dict(color='gray', width=2, dash='dash'),
                opacity= 0.8))
test_fig.add_trace(go.Scatter(
                x= prophet_pred.index,
                y= prophet_pred.yhat_upper,
                name= "Prediction Upper Bound",
                mode = 'lines',
                line = dict(color='royalblue', width=2, dash='dash'),
                opacity = 0.8
                ))

test_fig.update_layout(title_text= "Prophet Model's Test Prediction",
                       xaxis_title="Date", yaxis_title="Cases",)

test_fig.show()
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
 
metric_df = prophet_pred.set_index('ds')[['yhat']].join(time_series_data.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)
metric_df.tail()
r2_score(metric_df.y, metric_df.yhat)
mean_squared_error(metric_df.y, metric_df.yhat)
mean_absolute_error(metric_df.y, metric_df.yhat)

import warnings
warnings.filterwarnings('ignore')

import csv 
import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Bidirectional, RepeatVector, TimeDistributed, Input, Flatten, Dropout

from statsmodels.tools.eval_measures import rmse
from tabulate import tabulate
# prep data 

time_series_data = df_en[['Year', 'CardiovascularDiseases']].groupby('Year', as_index = False).sum()
time_series_data.columns = ['ds', 'y']
time_series_data.ds = pd.to_datetime(time_series_data.ds, format='%Y')
time_series_data.ds = pd.to_datetime(time_series_data.ds)
time_series_data.shape
data = time_series_data.set_index('ds')
train_data = data[:len(data)-3]
test_data = data[len(data)-3:]
scalar = MinMaxScaler()
scalar.fit(train_data)
scaled_train_data = scalar.transform(train_data)
scaled_test_data = scalar.transform(test_data)
steps = 3
n_features = 1 
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length = steps, batch_size=1)

model = Sequential()
model.add(LSTM(200, activation="relu", input_shape=(steps, n_features)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.summary()
model.fit_generator(generator, epochs=20)
losses_ = model.history.history["loss"]

len(losses_)
plt.figure(figsize=(15,4))
plt.xlabel = ("Epochs")
plt.ylabel = ("Loss")
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_)), losses_)
prediction_scaled = list()

batch = scaled_train_data[-steps:]
curr_batch = batch.reshape((1, steps, n_features))

for i in range(len(test_data)): 
    pred_ = model.predict(curr_batch)[0]
    prediction_scaled.append(pred_)
    curr_batch = np.append(curr_batch[:, 1:, :], [[pred_]], axis=1)
prediction_scaled

prediction_card = scalar.inverse_transform(prediction_scaled)
prediction_card
test_data["LSTM_Predictions"] = prediction_card
test_data
test_data['y'].plot(figsize = (15,4), legend=True)
test_data['LSTM_Predictions'].plot(legend = True);

lstm_rmse_error_card= rmse(test_data['y'], test_data["LSTM_Predictions"])
lstm_mse_error_card = lstm_rmse_error_card**2
mean_value = data['y'].mean()

print('MSE Error:', lstm_mse_error_card, '\nRMSE Error:', lstm_rmse_error_card, '\nMean:',mean_value)
model_bidirectional = Sequential()
model_bidirectional.add(Bidirectional(LSTM(200, activation='relu'), input_shape=(steps, n_features)))
model_bidirectional.add(Dense(1))
model_bidirectional.compile(optimizer='adam', loss='mse')
model_bidirectional.summary()

model_bidirectional.fit_generator(generator, epochs=20)

losses_bidirectional = model_bidirectional.history.history["loss"]


plt.figure(figsize=(15,4))
plt.xlabel = ("Epochs")
plt.ylabel = ("Loss")
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_bidirectional)), losses_bidirectional)
prediction_scaled_bidirectional = list()

batch = scaled_train_data[-steps:]
curr_batch = batch.reshape((1, steps, n_features))

for i in range(len(test_data)): 
    pred_bidirectional = model_bidirectional.predict(curr_batch)[0]
    prediction_scaled_bidirectional.append(pred_bidirectional)
    curr_batch = np.append(curr_batch[:, 1:, :], [[pred_bidirectional]], axis=1)
prediction_scaled_bidirectional
prediction_bidirectional = scalar.inverse_transform(prediction_scaled_bidirectional)
prediction_bidirectional
test_data["LSTM_Predictions"] = prediction_bidirectional
test_data
test_data['y'].plot(figsize = (15,4), legend=True)
test_data['LSTM_Predictions'].plot(legend = True);
lstm_rmse_error_bidirectional = rmse(test_data['y'], test_data["LSTM_Predictions"])
lstm_mse_error_bidirectional = lstm_rmse_error_bidirectional**2
mean_value = data['y'].mean()

print('MSE Error:', lstm_mse_error_bidirectional, '\nRMSE Error:', lstm_rmse_error_bidirectional, '\nMean:',mean_value)
