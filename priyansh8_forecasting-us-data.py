# Importing necessary libraries to conduct our analysis

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

# Ignore harmless warnings

import warnings

warnings.filterwarnings("ignore")

from IPython.display import HTML,display



warnings.filterwarnings("ignore")



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading the dataset into object 'df' using pandas:

import datetime

df= pd.read_csv('../input/final-1/19-20_pred.csv',parse_dates=True)



for i in range(19542):

    df['Date'][i] = datetime.datetime.strptime(df['Date'][i], "%d-%m-%Y").strftime("%Y-%m-%d")



df['Date'] = pd.to_datetime(df['Date'])
# df.drop(df[df['City'] =='Portland'].index, inplace = True)
df.shape
df.head(5)
df.describe()
# df=df[['City','Date','PM2.5','O3', 'CO']]
cities=pd.unique(df['City'])

column1= cities+'_PM2.5'

# column2=cities+'_O3'

# columns=[*column1,*column2]
len(column1)
final_df=pd.DataFrame(index=np.arange('2018-12-30','2020-05-30',dtype='datetime64[D]'),columns=column1)



print(final_df.shape)
arr=dict()

for i in range(len(cities)):

    arr[cities[i]] = 0

    



for i in range(len(cities)):

    for j in range(19542):

        if(cities[i]==df['City'][j]):

            arr[cities[i]]+=1

            

            

print(arr)
for city,i in zip(cities,final_df.columns):

    n=len(np.array(df[df['City']==city]['PM2.5']))

#     print(n)

    final_df[i][-n:]=np.array(df[df['City']==city]['PM2.5'])
final_df=final_df.astype('float64')

final_df=final_df.resample(rule='MS').mean()
final_df.tail()

# print(final_df.shape)
final_df['US_PM2.5']=final_df.mean(axis=1)
ax=final_df[['US_PM2.5']].plot(figsize=(12,8),grid=True,lw=2,color='Red')

ax.autoscale(enable=True, axis='both', tight=True)
df_2019=final_df['2019-01-01':'2020-01-01']

# print(df_2019.head())

# print(df_2019.shape)
# df_2019=final_df

# df_2019.head()

df_2019=df_2019.drop(['Brooklyn_PM2.5','Charlotte_PM2.5','Columbus_PM2.5','Detroit_PM2.5','Honolulu_PM2.5','Richmond_PM2.5','San Diego_PM2.5','Tallahassee_PM2.5','The Bronx_PM2.5'],axis=1)

# for col in df_2019.columns:

#     df_2019[col].fillna((df_2019[col].mean()), inplace=True)
df_2019.isna().sum()




# print(df_2019.head)

AQI_2019=df_2019.mean(axis=0)

AQI_2019.head()
plt.figure(figsize=(20,8))

plt.xticks(rotation=90)

bplot = sns.boxplot( data=df_2019,  width=0.75,palette="GnBu_d")

plt.ylabel('PM2.5');

bplot.grid(True)
plt.figure(figsize=(20,8))

plt.xticks(rotation=90)

plt.ylabel('PM2.5')

bplot=sns.barplot(AQI_2019.index, AQI_2019.values,palette="GnBu_d")

final_df.head()
from statsmodels.tsa.seasonal import seasonal_decompose

India_AQI=final_df['US_PM2.5']



print(India_AQI)

# result=seasonal_decompose(India_AQI,model='multiplicative')

# result.plot();
type(India_AQI)
# from matplotlib import dates

# ax=result.seasonal.plot(xlim=['2018-12-30','2020-05-15'],figsize=(20,8),lw=2)

# ax.yaxis.grid(True)

# ax.xaxis.grid(True)
# Load specific forecasting tools

from statsmodels.tsa.statespace.sarimax import SARIMAX

!pip install pmdarima;

from pmdarima import auto_arima;                              # for determining ARIMA orders
auto_arima(y=India_AQI,start_p=0,start_P=0,start_q=0,start_Q=0,seasonal=False, m=12).summary()
# len(India_AQI)
#dividing into train and test:

# train=India_AQI[:41]

# test=India_AQI[42:54]
# Forming the model:

model=SARIMAX(train,order=(1,1,1),seasonal_order=(1,0,1,12),)

results=model.fit()

results.summary()

#Obtaining predicted values:

predictions = results.predict(start=42, end=53, typ='levels')
#Plotting predicted values against the true values:

predictions.plot(legend=True)

test.plot(legend=True)
# from sklearn.metrics import mean_squared_error

# RMSE=np.sqrt(mean_squared_error(predictions,test))

# print('RMSE = ',RMSE)

# print('Mean AQI',test.mean())
#dividing into train and test:

# train=India_AQI[:53]

# test=India_AQI[54:]

# # Forming the model:

# model=SARIMAX(train,order=(1,1,1),seasonal_order=(1,0,1,12),)

# results=model.fit()

# results.summary()

# #Obtaining predicted values:

# predictions = results.predict(start=54, end=64, typ='levels').rename('Predictions')

# #Plotting predicted values against the true values:

# predictions.plot(legend=True)

# test.plot(legend=True);
#Finding RMSE:

# from sklearn.metrics import mean_squared_error

# RMSE=np.sqrt(mean_squared_error(predictions,test))

# print('RMSE = ',RMSE)

# print('Mean AQI',test.mean())
# Forming the model:

# model=SARIMAX(India_AQI,order=(1,1,1),seasonal_order=(1,0,1,12))

# results=model.fit()

# results.summary()

# #Obtaining predicted values:

# predictions = results.predict(start=64, end=77, typ='levels').rename('Predictions')

# #Plotting predicted values against the true values:

# predictions.plot(legend=True)

# India_AQI.plot(legend=True,figsize=(12,8),grid=True);
India_AQI = India_AQI.to_frame()
# India_AQI=India_AQI.set_index('ds')
India_AQI.shape
India_AQI.reset_index()

# India_AQI.columns = ['Date','US_PM2.5']

# print(type(India_AQI))
train=India_AQI[:-5]

test=India_AQI[-5:]

print(train)

print(test)

print(type(train))

df = train

# train.reshape(-1,1)

# test.reshape(-1,1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)

scaled_test = scaler.transform(test)
print(train.shape)
from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 10

n_features = 1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
#To give an idea of what generator file holds:

X,y = generator[0]
# We can see that the x array gives the list of values that we are going to predict y of:

print(f'Given the Array: \n{X.flatten()}')

print(f'Predict this y: \n {y}')
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.layers import LSTM

# from keras
# defining the model(note that  I am using a very basic model here, a 2 layer model only):

model = Sequential()

model.add(LSTM(50,activation='relu', input_shape=(n_input, n_features)))

# model.add(LSTM(50,return_sequences = True, activation='relu'))

# model.add(LSTM(32, activation='relu'))

# model.add(Dense(1))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse',metrics=['acc'])



model.summary()



# model = Sequential()

# model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))

# model.add(Dropout(0.15))

# model.add(Dense(1))

# optimizer = keras.optimizers.Adam(learning_rate=0.001)

# model.compile(optimizer=optimizer, loss='mse', metrics = ['acc'])

# history = model.fit_generator(generator,epochs=100,verbose=1)
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
# Fitting the model with the generator object:

model.fit_generator(generator,epochs=20,verbose = 1)
loss_per_epoch = model.history.history['loss']

plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
test_predictions = []



first_eval_batch = scaled_train[-n_input:]

current_batch = first_eval_batch.reshape((1, n_input, n_features))



for i in range(len(test)):

    

    

    current_pred = model.predict(current_batch)[0]

    

    

    test_predictions.append(current_pred) 

    

    

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test.plot(figsize=(12,8))

plt.plot(true_predictions)

from sklearn.metrics import mean_squared_error

RMSE=np.sqrt(mean_squared_error(test['US_PM2.5'],test['Predictions']))

print('RMSE = ',RMSE)

print('US_PM2.5=',India_AQI['US_PM2.5'].mean())
scaler.fit(India_AQI)

scaled_India_AQI=scaler.transform(India_AQI)
generator = TimeseriesGenerator(scaled_India_AQI, scaled_India_AQI, length=n_input, batch_size=1)
model.fit_generator(generator,epochs=40, verbose = 1)
test_predictions = []



first_eval_batch = scaled_India_AQI[-n_input:]

current_batch = first_eval_batch.reshape((1, n_input, n_features))



for i in range(len(test)):

    

    

    current_pred = model.predict(current_batch)[0]

    

    

    test_predictions.append(current_pred) 

    

    

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions=true_predictions.flatten()
true_preds=pd.DataFrame(true_predictions,columns=['Forecast'])

true_preds=true_preds.set_index(pd.date_range('2020-09-01',periods=5,freq='MS'))
true_preds
plt.figure(figsize=(20,8))

plt.grid(True)

plt.plot( true_preds['Forecast'])

plt.plot( India_AQI['US_PM2.5'])
print(train)

print(test)

df = train
# import numpy as np

# import pandas as pd

# from pandas.tseries.offsets import DateOffset

# from sklearn.preprocessing import MinMaxScaler



# import tensorflow as tf

# from tensorflow import keras



# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# from tensorflow.keras.models import Sequential

# from tensorflow.keras.layers import Dense

# from tensorflow.keras.layers import LSTM

# from tensorflow.keras.layers import Dropout

# import warnings

# warnings.filterwarnings("ignore")



# import chart_studio as py

# import plotly.offline as pyoff

# import plotly.graph_objs as go

# pyoff.init_notebook_mode(connected=True)



# # def parser(x):

# #     return pd.datetime.strptime('190'+x, '%Y-%m')



# # df = pd.read_csv('shampoo.csv', parse_dates=[0], index_col=0, date_parser=parser)

# # df.tail()



# # train = df



# scaler = MinMaxScaler()

# scaler.fit(train)

# train = scaler.transform(train)



# n_input = 12

# n_features = 1

# generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)



# model = Sequential()

# model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))

# model.add(Dropout(0.15))

# model.add(Dense(1))



# optimizer = keras.optimizers.Adam(learning_rate=0.001)

# model.compile(optimizer=optimizer, loss='mse',metrics = ['acc'])



# history = model.fit_generator(generator,epochs=100,verbose=1)



# hist = pd.DataFrame(history.history)

# hist['epoch'] = history.epoch



# plot_data = [

#     go.Scatter(

#         x=hist['epoch'],

#         y=hist['loss'],

#         name='loss'

#     )

    

# ]



# plot_layout = go.Layout(

#         title='Training loss'

#     )

# fig = go.Figure(data=plot_data, layout=plot_layout)

# pyoff.iplot(fig)
# pred_list = []



# batch = train[-n_input:].reshape((1, n_input, n_features))



# for i in range(n_input):   

#     pred_list.append(model.predict(batch)[0]) 

#     batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)





# add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,13) ]

# future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)



# df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),

#                           index=future_dates[-n_input:].index, columns=['Prediction'])



# df_proj = pd.concat([df,df_predict], axis=1)



# df_proj.tail(12)



# plot_data = [

#     go.Scatter(

#         x=df_proj.index,

#         y=df_proj['US_PM2.5'],

#         name='actual'

#     ),

#     go.Scatter(

#         x=df_proj.index,

#         y=df_proj['Prediction'],

#         name='prediction'

#     )

# ]



# plot_layout = go.Layout(

#         title='Shampoo sales prediction'

#     )

# fig = go.Figure(data=plot_data, layout=plot_layout)

# pyoff.iplot(fig)
# pip install chart_studio
# from tensorflow import keras
# conda install -c plotly chart-studio
# pip install chart-studio
# print(df)
# test_predictions = []



# first_eval_batch = scaled_train[-n_input:]

# current_batch = first_eval_batch.reshape((1, n_input, n_features))



# for i in range(len(test)):

    

    

#     current_pred = model.predict(current_batch)[0]

    

    

#     test_predictions.append(current_pred) 

    

    

#     current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)