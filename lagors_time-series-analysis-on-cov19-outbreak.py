import pandas as pd

import numpy as np

import itertools

import datetime 

#manipulating data and basic python libraries



import sklearn

from sklearn.metrics import mean_absolute_error

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

#Preprocessing data and metrics for error evaluation



import tensorflow as tf

import tensorflow.keras as keras

from keras.models import Sequential

from keras.layers import Dense, Dropout,Activation, LSTM, ReLU

from keras.optimizers import Adam

from keras.callbacks import callbacks, EarlyStopping, ReduceLROnPlateau

#Tensorflow as our deep learning back end



from statsmodels.tsa.arima_model import ARIMA

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Statistical modeling using Arima



from fbprophet import Prophet

#Facebook prophet modeling



import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

sns.set(palette = 'Set2',style='dark',font=['simhei', 'Arial'])

#Visualization



import warnings

warnings.simplefilter('ignore')

#Ignoring warnings
data=pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')



data['Date']=pd.to_datetime(data['Date'])

data= data.rename(columns={'Country/Region':'Country','Date':'ds','Province/State':'Province'}) 

data.replace('Korea, South','South Korea',inplace=True)

data.head()
df=data.groupby(['ds','Country']).agg('sum').reset_index()

df['date']=df['ds']

for i in range (0,len(df)):

    df['date'][i]=df['date'][i].strftime("%d %b")

df.tail(5)
dfCountry=df.groupby(['Country']).agg('max')

dfCountry.sort_values('Confirmed',ascending=False,inplace=True)

dfCountry=dfCountry[0:10]

dfCountry.reset_index(inplace=True)



f=plt.figure(figsize=(13,7))

ax=sns.barplot(x=dfCountry['Country'],y=dfCountry['Confirmed'], color='Orange',alpha=0.8,label='Confirmed cases')

ax=sns.barplot(x=dfCountry['Country'],y=dfCountry['Recovered']+dfCountry['Deaths'],alpha=0.8, color='purple', label='Terminated cases')

plt.legend()

plt.ylabel('')

plt.xticks(rotation=25)

a=ax.set_title('Cases by country')
dfus=df.groupby(['Country']).agg('max')

dfus.sort_values('Confirmed',ascending=False,inplace=True)

dfus.reset_index(inplace=True)

row_df = pd.DataFrame(dfus.loc[0]).T.drop(columns=['Lat','Long','ds','date'])

dfus=dfus[1:].drop(columns=['Lat','Long','ds','date']).sum(axis=0)

dfus=pd.DataFrame([dfus])

dfus = pd.concat([row_df, dfus])

dfus.reset_index(inplace=True,drop=True)

dfus.at[1, 'Country'] = 'Rest of the world'

f=plt.figure(figsize=(5,7))

ax=sns.barplot(x=dfus['Country'],y=dfus['Confirmed'], color='Orange',alpha=0.8,label='Confirmed cases')

ax=sns.barplot(x=dfus['Country'],y=dfus['Recovered']+dfus['Deaths'],alpha=0.8, color='purple', label='Terminated cases')

plt.legend()

plt.ylabel('')

a=ax.set_title('Cases by country')
dfCountry=dfCountry[1:]

f=plt.figure(figsize=(13,7))

ax=sns.barplot(x=dfCountry['Country'],y=dfCountry['Confirmed'], color='Orange',alpha=0.8,label='Confirmed cases')

ax=sns.barplot(x=dfCountry['Country'],y=dfCountry['Recovered']+dfCountry['Deaths'],alpha=0.8, color='purple', label='Terminated cases')

plt.legend()

plt.ylabel('')

plt.xticks(rotation=25)

a=ax.set_title('Cases by country')
img=plt.imread('../input/wmappicture/WMap.png')



dfmap=df.loc[(df['ds']==df['ds'].max())]

dfmap.drop(columns=['Lat','Long'],inplace=True)

dfmap=dfa= pd.merge(dfmap, data, on=['Country','ds'], how='inner')



bins=[0,100,1000,10000,100000,1000000,100000000]

labels=['1-100','100-1000','1000-10000','10k-100k','100k-1m','1m+']

dfmap['Category']= pd.cut(dfmap["Confirmed_y"], bins , labels=labels)



f,ax=plt.subplots(figsize=(25,18))



ax.imshow(img, zorder=0,extent=( -174,188,-60, 120))

ax=plt.gca()



dfmap=dfmap.loc[dfmap['Confirmed_y']>1]

dfmap['Confirmed_y']=100*np.log(dfmap['Confirmed_y'])

dfmap.sort_values(by='Confirmed_y', inplace=True, ascending=True)



g=sns.scatterplot(y=dfmap['Lat'],x=dfmap['Long'], hue=dfmap['Category'],legend='full',s=dfmap['Confirmed_y']*2)

a=g.set(xlabel='', ylabel='', yticklabels='',xticklabels='', title='Distribution of confirmed cases around the world')

del(dfmap)
sns.set(palette = 'Dark2',style='darkgrid')



def pltCountry(case,*argv):

    f, ax=plt.subplots(figsize=(16,5))

    labels=argv

    for a in argv: 

        country=df.loc[(df['Country']==a)]

        country.reset_index(inplace=True,drop=True)      

        plt.plot(country['date'],country[case],linewidth=3)

        plt.xticks(rotation=40)

        plt.legend(labels)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))

        ax.set(title='Evolution of the number of %s cases'%case,xlabel='Date',ylabel='Number of %s cases'%case )

        

        

pltCountry('Confirmed','China')
pltCountry('Confirmed', 'Germany','Spain','France','US')

pltCountry('Deaths','Germany','Spain','France','US')

pltCountry('Recovered', 'Germany','Spain','France','US')
pltCountry('Confirmed','South Korea','Italy','Iran')

pltCountry('Deaths','South Korea','Italy','Iran')

pltCountry('Recovered','South Korea','Italy','Iran')
df['Actual']=df['Confirmed']-(df['Deaths']+df['Recovered'])



def pltCountryDet(*argv):

    f,ax=plt.subplots(figsize=(16,5))

    labels=argv

    cases=['Confirmed','Deaths','Recovered']

    for case in cases:

        for a in argv: 

            country=df.loc[(df['Country']==a)]

            plt.plot(country['date'],country[case],linewidth=2)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(7))

            plt.xticks(rotation=40)

            plt.legend(cases)

            ax.set(title='Evolution in %s '%a,xlabel='Date',ylabel='Number of cases' )

            

    

pltCountryDet('Germany') 
def actualCases(*argv):

    f,ax=plt.subplots(figsize=(16,5))

    labels=argv  

    

    for a in argv:

        country=df.loc[(df['Country']==a)]

        plt.plot(country['date'],country['Actual'],linewidth=2)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))

        plt.xticks(rotation=40)

        plt.legend(labels)

        ax.set(title=' Evolution of actual cases',xlabel='Date',ylabel='Number of cases' )





actualCases('France','Italy','Germany')
case='Confirmed'

def timeCompare(time,*argv):

    Coun1=argv[0]

    Coun2=argv[1]

    f,ax=plt.subplots(figsize=(16,5))

    labels=argv  

    country=df.loc[(df['Country']==Coun1)]

    plt.plot(country['ds'],country[case],linewidth=2)

    plt.xticks([])

    plt.legend(labels)

    ax.set(title=' Evolution of actual cases',ylabel='Number of cases' )



    country2=df.loc[df['Country']==Coun2]

    country2['ds']=country2['ds']-datetime.timedelta(days=time)

    plt.plot(country2['ds'],country2[case],linewidth=2)

    plt.xticks([])

    plt.legend(labels)

    ax.set(title=' Evolution of cases with %d days difference '%time ,ylabel='Number of %s cases'%case )

    



timeCompare(8,'Italy','France')

timeCompare(5,'Italy','Spain')

timeCompare(7,'Italy','Germany')

timeCompare(7,'Italy','US')
case='Actual'

timeCompare(40,'China','Italy')

country=df.loc[(df['Country']=='Italy')]

f=plt.figure(figsize=(20,7))

country['Confirmed']= country['Confirmed']-country['Confirmed'].shift(1)

country['Deaths']= country['Deaths']-country['Deaths'].shift(1)

country['Recovered']= country['Recovered']-country['Recovered'].shift(1)

country['ds']=pd.to_datetime(country['ds'],format='%Y%m%d')

country=country[32:]

country.reset_index(inplace=True,drop=True)

country['date']=country['ds']

for i in range (0,len(country)):

    country['date'][i]=country['date'][i].strftime("%d %b")

country.drop(index=18,inplace=True) #The march 12th data is not provided in the dataset

ax=sns.barplot(x=country['date'],y=country['Confirmed'], color='Orange',alpha=0.8,label='Confirmed cases')

ax=sns.barplot(x=country['date'],y=country['Recovered']+country['Deaths'],alpha=0.8, color='purple', label='Terminated cases')

plt.legend()

plt.ylabel('')

plt.xticks(rotation=-90)

plt.xlabel('')

a=ax.set_title('Evolution of daily cases in Italy per day starting on 23rd Feb (reached 100 confirmed cases)')
country=df.loc[(df['Country']=='US')]

f=plt.figure(figsize=(20,7))

country['Confirmed']= country['Confirmed']-country['Confirmed'].shift(1)

country['Deaths']= country['Deaths']-country['Deaths'].shift(1)

country['Recovered']= country['Recovered']-country['Recovered'].shift(1)

country['ds']=pd.to_datetime(country['ds'],format='%Y%m%d')

country=country[40:]

country.reset_index(inplace=True,drop=True)

country['date']=country['ds']

for i in range (0,len(country)):

    country['date'][i]=country['date'][i].strftime("%d %b")

ax=sns.barplot(x=country['date'],y=country['Confirmed'], color='Orange',alpha=0.8,label='Confirmed cases')

ax=sns.barplot(x=country['date'],y=country['Recovered']+country['Deaths'],alpha=0.8, color='purple', label='Terminated cases')

plt.legend()

plt.ylabel('')

plt.xticks(rotation=-90)

plt.xlabel('')

a=ax.set_title('Evolution of daily cases in the US per day starting on March 2nd (reached 100 confirmed cases)')
#sns.set(palette = 'Set1',style='darkgrid')

#Function for making a time serie on a designated country and plotting the rolled mean and standard 

def roll(country,case='Confirmed'):

    ts=df.loc[(df['Country']==country)]  

    ts=ts[['ds',case]]

    ts=ts.set_index('ds')

    ts.astype('int64')

    a=len(ts.loc[(ts['Confirmed']>=10)])

    ts=ts[-a:]

    return (ts.rolling(window=4,center=False).mean().dropna())





def rollPlot(country, case='Confirmed'):

    ts=df.loc[(df['Country']==country)]  

    ts=ts[['ds',case]]

    ts=ts.set_index('ds')

    ts.astype('int64')

    a=len(ts.loc[(ts['Confirmed']>=10)])

    ts=ts[-a:]

    plt.figure(figsize=(16,6))

    plt.plot(ts.rolling(window=7,center=False).mean().dropna(),label='Rolling Mean')

    plt.plot(ts[case])

    plt.plot(ts.rolling(window=7,center=False).std(),label='Rolling std')

    plt.legend()

    plt.title('Cases distribution in %s with rolling mean and standard' %country)

    plt.xticks([])



tsC=roll('China')

rollPlot('China')
#Decomposing the ts to find its properties

fig=sm.tsa.seasonal_decompose(tsC.values,freq=7).plot()

#Function to check the stationarity of the time serie using Dickey fuller test

def stationarity(ts):

    print('Results of Dickey-Fuller Test:')

    test = adfuller(ts, autolag='AIC')

    results = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for i,val in test[4].items():

        results['Critical Value (%s)'%i] = val

    print (results)



tsC=tsC['Confirmed'].values

stationarity(tsC)


#Differenciating every elemenet from the previous element to make ts stationary

#def difference(ts, times=1):

#    diff = list()

#    for i in range(times, len(ts)):

#        val = ts[i] - ts[i - times]

#        diff.append(val)

#    return pd.Series(diff)





#print('\n\nDickey fuller test after de-trend \n')

#tsCD=difference(tsC,1)

#tsCD=tsCD.values 

#plt.plot(tsCD)

#stationarity(tsCD)
#auto-Correlation and partial auto-correlation plots

def corr(ts):

    plot_acf(ts,lags=12,title="ACF")

    plot_pacf(ts,lags=12,title="PACF")

    

corr(tsC)
#Mean absolute percentage error

def mape(y, y_pred): 

    y, y_pred = np.array(y), np.array(y_pred)

    return np.mean(np.abs((y - y_pred) / y)) * 100



def split(ts):

    #splitting 80%/20% because of little amount of data

    size = int(len(ts) * 0.80)

    train= ts[:size]

    test = ts[size:]

    return(train,test)





#Arima modeling for ts

def arima(ts,test):

    p=d=q=range(0,6)

    a=99999

    pdq=list(itertools.product(p,d,q))

    

    #Determining the best parameters

    for var in pdq:

        try:

            model = ARIMA(ts, order=var)

            result = model.fit()



            if (result.aic<=a) :

                a=result.aic

                param=var

        except:

            continue

            

    #Modeling

    model = ARIMA(ts, order=param)

    result = model.fit()

    result.plot_predict(start=int(len(ts) * 0.7), end=int(len(ts) * 1.2))

    pred=result.forecast(steps=len(test))[0]

    #Plotting results

    f,ax=plt.subplots()

    plt.plot(pred,c='green', label= 'predictions')

    plt.plot(test, c='red',label='real values')

    plt.legend()

    plt.title('True vs predicted values')

    #Printing the error metrics

    print(result.summary())        

    print('\nMean absolute error: %f'%mean_absolute_error(test,pred))

    print('\nMean absolute percentage error: %f'%mape(test,pred))

    return (pred)







train,test=split(tsC)

pred=arima(train,test)

#Taking the US example

tsU=roll('US')

rollPlot('US')



a=len(tsU.loc[(tsU['Confirmed']>=10)])

tsU=tsU[-a:]            #we omit the data in earliest stages for irrelevancy 



#renaming and preparing data for prophet

tsU.rename(columns={'Confirmed':'y'},inplace=True) #prophet requires the columns to be named ds and y

train,test= split(tsU) 



train.reset_index(inplace=True)

test.reset_index(inplace=True)
#We fit the data and extract the predictions

model=Prophet(daily_seasonality=True)

model.fit(train)



pred = model.predict(test)

a=model.plot_components(pred)
f,ax =plt.subplots(figsize=(16,5))

plt.plot(test['ds'],test['y'],label='True values',c='red',marker='.')

plt.title('True and predicted values')

a=model.plot(pred, ax=ax)

ax.legend()



print('Mean absolute error: %d'%mean_absolute_error(test['y'],pred['yhat']))

print('Mean absolute  percentage error: %d'%mape(test['y'],pred['yhat']))
#In this example, prophet didn't do a good job, we notice how the data is inconsistently growing in this #example, we migh think prophet doesn't perform well with this kind of linear distrubtion. A regression model #might be more accurate for this prediction.
def proph(country,case='Confirmed'):

    tsU=roll(country,case)

    tsU.rename(columns={case:'y'},inplace=True) 

    tsU=tsU[30:] 

    train,test= split(tsU)



    train.reset_index(inplace=True)

    test.reset_index(inplace=True)



    model=Prophet(daily_seasonality=True)

    model.fit(train)



    pred = model.predict(test)

    f,ax =plt.subplots(figsize=(16,5))

    plt.plot(test['ds'],test['y'],label='True values',c='red',marker='.')

    plt.title('True and predicted values')

    a=model.plot(pred, ax=ax)

    ax.legend()



    print('Mean absolute error: %f'%mean_absolute_error(test['y'],pred['yhat']))

    print('Mean absolute  percentage error: %f'%mape(test['y'],pred['yhat']))

    return(pred)
tsI=roll('Italy')

rollPlot('Italy')

#Scale and split the data

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(tsI['Confirmed'].values.reshape(-1,1).astype('float32'))



train,test=split(scaled)
#Transform our ts to a usable df with lookback 

#This means we make our ts a supervised learning modelizable df

def lookBack(ts, look_back=2):

    X=list()

    Y=list()

    for i in range(len(ts) - look_back):

        a = ts[i:(i + look_back), 0]

        X.append(a)

        Y.append(ts[i + look_back, 0])

    return np.array(X), np.array(Y)



trainX, trainY = lookBack(train)

testX, testY = lookBack(test)



#Reshape our data

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Modeling and fitting

model = Sequential()



model.add(LSTM(1028, dropout=3, input_shape=(1,2), activation='relu'))

model.add(Dense(1))



early_stop=EarlyStopping(monitor='val_loss', patience=10, verbose=0)

lr_reduce=ReduceLROnPlateau(monitor='val-loss',patience=5)

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

history=model.fit(trainX,trainY, epochs = 100, batch_size = 12,validation_data=(testX, testY), 

                  shuffle=False, callbacks=[early_stop,lr_reduce])

model.summary()
#plotting the error reduction

plt.subplots(figsize=(13,5))

plt.plot(history.history['loss'], label='Train')

plt.plot(history.history['val_loss'], label='Test')

plt.title('Error reduction')

a=plt.legend()
#plotting our resutls and the error of our scaled data

pred=model.predict(testX)

f,ax=plt.subplots(figsize=(13,5))

plt.plot(pred, label='Predict')

plt.plot(testY, label='True')

plt.legend()

plt.title('Predicted vs true scaled values')



print('Mean absolute error: %f'%mean_absolute_error(testY,pred))

print('Mean absolute  percentage error: %f'%mape(testY,pred))
#reversing the scaling

predictions= scaler.inverse_transform(pred.reshape(-1, 1))

test_val = scaler.inverse_transform(testY.reshape(-1, 1))

#getting the dates back

dates = tsI.tail(len(testX)).index

#reshaping

predictions = predictions.reshape(len(predictions))

test_val = test_val.reshape(len(test_val))



#plotting the data

f, ax=plt.subplots(figsize=(16,5))

plt.plot(tsI)

plt.plot(dates, test_val, label= 'True', c='red',marker='.')

plt.plot(dates, predictions, label= 'Predicted', marker='.' ,c='purple')

plt.xticks([])

plt.title('True vs predicted values')

a=plt.legend()

print('Mean absolute error: %f'%mean_absolute_error(test_val,predictions))

print('Mean absolute  percentage error: %f'%mape(test_val,predictions))
def lstm(country,case='Confirmed'):

    ts=roll(country,case)

    

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled = scaler.fit_transform(ts[case].values.reshape(-1,1).astype('float32'))



    train,test=split(scaled)



    trainX, trainY = lookBack(train)

    testX, testY = lookBack(test)



    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    

    model = Sequential()



    model.add(LSTM(1028, dropout=3, input_shape=(1,2), activation='relu'))

    model.add(Dense(1))



    early_stop=EarlyStopping(monitor='val_loss', patience=10, verbose=0)

    lr_reduce=ReduceLROnPlateau(monitor='val-loss',patience=5)

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    history=model.fit(trainX,trainY, epochs = 100, batch_size = 12,validation_data=(testX, testY), 

                      shuffle=False, callbacks=[early_stop,lr_reduce])

    

    pred=model.predict(testX)

    

    predictions= scaler.inverse_transform(pred.reshape(-1, 1))

    test_val = scaler.inverse_transform(testY.reshape(-1, 1))



    dates = ts.tail(len(testX)).index



    predictions = predictions.reshape(len(predictions))

    test_val = test_val.reshape(len(test_val))



    f, ax=plt.subplots(figsize=(16,5))

    plt.plot(ts)

    plt.plot(dates, test_val, label= 'True', c='red',marker='.')

    plt.plot(dates, predictions, label= 'Predicted', marker='.' ,c='purple')

    plt.xticks([])

    plt.title('True vs predicted values')

    a=plt.legend()

    print('Mean absolute error: %f'%mean_absolute_error(test_val,predictions))

    print('Mean absolute  percentage error: %f'%mape(test_val,predictions))

    return predictions

rollPlot('Germany')
ts=roll('Germany')

ts=ts['Confirmed'].values

train,test=split(ts)

a=arima(train,test)
a=proph('Germany')
a=lstm('Germany')