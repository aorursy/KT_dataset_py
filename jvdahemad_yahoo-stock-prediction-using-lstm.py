import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import seaborn as sns

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler 

import pylab
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

plt.rcParams["figure.figsize"] = (7.5,5)
plt.style.use('seaborn')

pylab.rc('figure', figsize=(10,7))

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
stock=pd.read_csv('/kaggle/input/YahooFinance_Stock_2014-2019_train.csv',header=0,index_col=0,parse_dates=True,squeeze=True)
stock.head()
stock[["Open","High","Low","Close"]].plot.area(figsize=(15,10),alpha=0.5);
plt.title('Yahoo Finance Stock trend (2014-2019)')
plt.show()
stock=stock["Open"]
##Grouping the data by Year
groups=stock.groupby(pd.Grouper(freq='A'))

years={}
for name, group in groups:
    years[name.year] = group.values
    
years[2017]=np.append(years[2017],years[2017][-1]) ## Keeping the number of days same. Just to visualise the yearly data
years[2018]=np.append(years[2018],years[2018][-1]) 

years=pd.DataFrame(years)
years.plot.box(figsize=(15,10))
plt.title('Stock Price distribution by year')
plt.ylabel('Stock Price')
plt.xlabel('Year')
plt.show()
plt.figure(figsize=(15,7.5))
sns.barplot(years.iloc[-1,:].index,years.iloc[-1,:].values,palette='YlOrBr')
plt.title('End of year Stock Value')
plt.ylabel('Stock Price')
plt.xlabel('Year')
plt.show()
years.plot(figsize=(15,7.5))
plt.title('Yahoo Stock Trends by year',fontsize=20)
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.show()
fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(15,15))
pd.plotting.lag_plot(years[2014],ax=ax[0,0],lag=1)
pd.plotting.lag_plot(years[2015],ax=ax[0,1],lag=1)
pd.plotting.lag_plot(years[2016],ax=ax[1,0],lag=1)
pd.plotting.lag_plot(years[2017],ax=ax[1,1],lag=1)
pd.plotting.lag_plot(years[2018],ax=ax[2,0],lag=1)
pd.plotting.lag_plot(years[2019],ax=ax[2,1],lag=1)
fig.suptitle('1 Day Lag Plots',fontsize=20)
ax[0,0].set_title('2014',X=0.1)
ax[0,1].set_title('2015',X=0.1)
ax[1,0].set_title('2016',X=0.1)
ax[1,1].set_title('2017',X=0.1)
ax[2,0].set_title('2018',X=0.1)
ax[2,1].set_title('2019',X=0.1)
plt.show()
fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(15,15))
pd.plotting.autocorrelation_plot(years[2014],ax=ax[0,0])
pd.plotting.autocorrelation_plot(years[2015],ax=ax[0,1])
pd.plotting.autocorrelation_plot(years[2016],ax=ax[1,0])
pd.plotting.autocorrelation_plot(years[2017],ax=ax[1,1])
pd.plotting.autocorrelation_plot(years[2018],ax=ax[2,0])
pd.plotting.autocorrelation_plot(years[2019],ax=ax[2,1])
fig.suptitle('Auto corelation Plots',fontsize=20)
ax[0,0].set_title('2014',X=0.1)
ax[0,1].set_title('2015',X=0.1)
ax[1,0].set_title('2016',X=0.1)
ax[1,1].set_title('2017',X=0.1)
ax[2,0].set_title('2018',X=0.1)
ax[2,1].set_title('2019',X=0.1)
#plt.rc('axes', titlesize=20)
plt.show()
fig,ax=plt.subplots(nrows=3,ncols=1,figsize=(12,12))
stock.rolling(window=1).mean().plot(ax=ax[0])
stock.rolling(window=7).mean().plot(ax=ax[1])
stock.rolling(window=30).mean().plot(ax=ax[2])
#stock.rolling(window=7).mean().plot(ax=ax[1,1])
#fig.suptitle('Auto corelation Plots',fontsize=20)
ax[0].set_title('Daily Moving Average',fontsize=20)
ax[1].set_title('7-Days Moving Average',fontsize=20)
ax[2].set_title('30-Days Moving Average',fontsize=20)
#ax[1,1].set_title('2016')
plt.tight_layout()
data_train=stock.reset_index()
data_train.columns=['ds','y']
model=Prophet()
model.fit(data_train) ##Fitting our data

future=model.make_future_dataframe(periods=365)
predict=model.predict(future)
fig1=model.plot(predict,figsize=(12,7.5))
plt.title('Yahoo Stock Trends',fontsize=20)
plt.ylabel('Stock Price')
plt.xlabel('Year')
fig = model.plot(predict,figsize=(12,7.5))
a = add_changepoints_to_plot(fig.gca(), model, predict)
plt.title('Yahoo Stock Trends with Potential change points',fontsize=20)
plt.ylabel('Stock Price')
plt.xlabel('Year')
fig2=model.plot_components(predict,figsize=(12,10))
training_dataset=pd.read_csv('/kaggle/input/YahooFinance_Stock_2014-2019_train.csv')
training_data=training_dataset.iloc[:,1:2].values


sc=MinMaxScaler(feature_range=(0,1))  ##Normalising
training_data=sc.fit_transform(training_data)

#Scaling

X_train=[]
y_train=[]
for i in range(60,len(training_data)):
    X_train.append(training_data[i-60:i,0])
    y_train.append(training_data[i,0])

X_train= np.array(X_train)
y_train=np.array(y_train)    

X_train=X_train.reshape((len(training_data)-60),60,1)
regressor= Sequential()

#Adding LSTM layers

regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

#Compiling RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.summary()
regressor.fit(X_train,y_train,epochs=100,batch_size=32)
predict_2020=predict[(predict['ds']>'2019-12-31') & (predict['ds']<'2020-8-01')][['ds','trend']]  ##Extracting 2020 prediction from linear Model
predict_2020=predict_2020.set_index('ds',drop=True)

Xt=test_dataset.set_index('Date',drop=True)  ##Ignoring the holidays
new=Xt.join(predict_2020)
linear_prediction=new['trend'].values.reshape(len(new),1)
#Getting the real stock price of 2020
test_dataset=pd.read_csv('/kaggle/input/YahooFinance_Stock_2020_Test.csv')
real_stock_prices=test_dataset.iloc[:,1:2].values
total_dataset=pd.concat((training_dataset['Open'],test_dataset['Open']),axis=0)
inputs=total_dataset.iloc[len(total_dataset)-len(test_dataset)-60:].values
inputs=inputs.reshape(-1,1)

inputs=sc.transform(inputs)

X_test=[]
for i in range(60,(60+len(real_stock_prices))):
    X_test.append(inputs[i-60:i,0])
    
X_test= np.array(X_test)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

predicted_stock_values=regressor.predict(X_test)
predicted_stock_values=sc.inverse_transform(predicted_stock_values)
Year_2020=pd.DataFrame(np.concatenate((real_stock_prices,predicted_stock_values,linear_prediction),axis=1),index=test_dataset['Date'],columns=['Real','Predicted','Linear'])
Year_2020.index = pd.to_datetime(Year_2020.index)
Year_2020.tail()
groups_real = Year_2020['Real'].groupby(pd.Grouper(freq='M'))
groups_predict = Year_2020['Predicted'].groupby(pd.Grouper(freq='M'))

months_real = pd.concat([pd.DataFrame(x[1].values) for x in groups_real], axis=1)
months_predict = pd.concat([pd.DataFrame(x[1].values) for x in groups_predict], axis=1)

months_real = pd.DataFrame(months_real)
months_predict = pd.DataFrame(months_predict)

months_real.columns = ['Jan','Feb','Mar','Apr','May','Jun','July']
months_predict.columns = ['Jan','Feb','Mar','Apr','May','Jun','July']
plt.figure(figsize=(12,7.5))
plt.plot(real_stock_prices,color='red',label='Real Stock Price')
plt.plot(predicted_stock_values,color='blue',label='Predicted Stock Price')
#plt.plot(predicted_stock_values_2,color='green',label='Predicted Google Stock Price_2')
plt.plot(linear_prediction,color='black',label='Linear Prediction')
plt.title('January 2020 Stock Price of Yahoo')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()
mon=pd.DataFrame({'Real':months_real[18:19].values.ravel(),'predict':months_predict[18:19].values.ravel()},index=months_real[18:19].columns)
mon.plot(kind='bar',figsize=(12,7.5))
plt.xlabel("Month")
plt.ylabel("Stock Price")
plt.title("Month End Stock Price for 2020")
plt.show()