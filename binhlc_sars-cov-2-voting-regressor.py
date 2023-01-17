# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

train['Province/State'].fillna('', inplace=True)

test['Province/State'].fillna('', inplace=True)

train['Date'] =  pd.to_datetime(train['Date'])

test['Date'] =  pd.to_datetime(test['Date'])

train = train.sort_values(['Country/Region','Province/State','Date'])

test = test.sort_values(['Country/Region','Province/State','Date'])
# top country

train.groupby(['Country/Region']).max().sort_values(['ConfirmedCases'],ascending = False).head(10)
!pip install pmdarima
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import mean_squared_error

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



country_plot = 'Vietnam'



def RMSLE(pred,actual):

    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))



def RMSLE_Cal(pred_data):

    df = pd.DataFrame(pred_data)

    df.columns = ['Date','Country/Region','Province/State','ForecastId','ConfirmedCases','Fatalities']

    df_val = pd.merge(df,train[['Date','Country/Region','Province/State','ConfirmedCases','Fatalities']],on=['Date','Country/Region','Province/State'], how='left')

    df_val = df_val.sort_values('Date')

    val_size = df_val[df_val['ConfirmedCases_y'].notnull() == True].shape[0]

    return RMSLE(df_val['ConfirmedCases_y'][0:val_size-1].values, df_val['ConfirmedCases_x'][0:val_size-1].values)

    

def plotPrediction(pred_data):

    df = pd.DataFrame(pred_data)

    df.columns = ['Date','Country/Region','Province/State','ForecastId','ConfirmedCases','Fatalities']

    # Fix code: plot and caclulate only for Viet Nam

    df = df[df['Country/Region'] == country_plot]

    df_val = pd.merge(df,train[['Date','Country/Region','Province/State','ConfirmedCases','Fatalities']],on=['Date','Country/Region','Province/State'], how='left')

    df_val = df_val.sort_values('Date')

    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(20, 7))

    axes[0].plot(df_val['Date'], df_val['ConfirmedCases_y'], label = 'Confirmed Cases')

    axes[0].plot(df_val['Date'], df_val['ConfirmedCases_x'], label = 'Confirmed Cases Forecast')

    axes[1].plot(df_val['Date'], df_val['Fatalities_y'], label = 'Fatalities')

    axes[1].plot(df_val['Date'], df_val['Fatalities_x'], label = 'Fatalities Forecast')

    axes[0].legend()

    axes[1].legend()

    fig.autofmt_xdate()

    

    #return mean_squared_error(df_val['ConfirmedCases_y'][0:val_size-1], df_val['ConfirmedCases_x'][0:val_size-1])

    return RMSLE(df_val['ConfirmedCases_y'][0:val_size-1].values, df_val['ConfirmedCases_x'][0:val_size-1].values)

    
import pmdarima as pm

#import statsmodels.tsa.arima_model as ARIMA



pred_data = []

for country in train['Country/Region'].unique():

#for country in [country_plot]:

    for province in train[(train['Country/Region'] == country)]['Province/State'].unique():

        for forcast in test[(test['Country/Region'] == country) & (test['Province/State'] == province)]['ForecastId'].unique():

        #for forcast in [test[(test['Country/Region'] == country) & (test['Province/State'] == province)].max()['ForecastId']]:

            testdate = test[(test['Country/Region'] == country) & (test['Province/State'] == province) & (test['ForecastId'] == forcast)]['Date'].max()

            X_train = train[(train['Country/Region'] == country) & (train['Province/State'] == province) & (train['ConfirmedCases'] > 0) & (train['Date']<testdate)]['ConfirmedCases'].values

            if (len(X_train) < 1):

                ConfirmedCases_hat = 0

            elif (len(X_train) < 5):

                ConfirmedCases_hat = X_train[-1]

            elif((X_train[-1] == X_train[-2]) & (X_train[-2] == X_train[-3])):

                ConfirmedCases_hat = X_train[-1]

            else:

                #if (len(X_train) < 30):

                #    model = ARIMA.ARMA(X_train, order=(1,0,0))

                #else:

                #    model = ARIMA.ARMA(X_train, order=(3,0,3))

                

                if (testdate <= train[(train['Country/Region'] == country) & (train['Province/State'] == province)]['Date'].max()):

                    # Only train one time when have not train data to update

                    model_c = pm.auto_arima(X_train, suppress_warnings=True, seasonal=False, error_action="ignore")

                    #model = pm.auto_arima(X_train, suppress_warnings=True, seasonal=False, error_action="ignore")

                    #ConfirmedCases_hat = model.fit(disp=0).predict(start = len(X_train), end = len(X_train)+1)[0]

                    ConfirmedCases_hat = model_c.predict(n_periods=1)[-1]

                else:

                    n_period = testdate - train[(train['Country/Region'] == country) & (train['Province/State'] == province)]['Date'].max()

                    #ConfirmedCases_hat = model.fit(disp=0).predict(start = 0, end = len(X_train) + n_period.days)[-1]

                    ConfirmedCases_hat = model_c.predict(n_periods=n_period.days + 1)[-1]

                # For big change in a day  -- Andorra case  

                if (ConfirmedCases_hat < X_train[-1]):

                    ConfirmedCases_hat = X_train[-1]

                    

            # Don't overlap train and test data                    

            X_train = train[(train['Country/Region'] == country) & (train['Province/State'] == province) & (train['Fatalities'] > 0) & (train['Date']<testdate)]['Fatalities'].values

            if (len(X_train) < 1):

                Fatalities_hat = 0

            elif (len(X_train) < 5):

                Fatalities_hat = X_train[-1]

            elif((X_train[-1] == X_train[-2]) & (X_train[-2] == X_train[-3])):

                Fatalities_hat = X_train[-1]

            else:            

                #if (len(X_train) < 30):

                #    model = ARIMA.ARMA(X_train, order=(1,0,0))

                #else:

                #    model = ARIMA.ARMA(X_train, order=(3,0,3))

                

                if (testdate < train[(train['Country/Region'] == country) & (train['Province/State'] == province)]['Date'].max()):

                    model_f = pm.auto_arima(X_train, suppress_warnings=True, seasonal=False, error_action="ignore")

                    #Fatalities_hat = model.fit(disp=0).predict(start = 0, end = len(X_train)+1)[-1]

                    Fatalities_hat = model_f.predict(n_periods=1)[-1]

                else:

                    n_period = testdate - train[(train['Country/Region'] == country) & (train['Province/State'] == province)]['Date'].max()

                    #Fatalities_hat = model.fit(disp=0).predict(start = 0, end = len(X_train) + n_period.days)[-1]

                    Fatalities_hat = model_f.predict(n_periods=n_period.days + 1)[-1]

                # For big change in a day  -- Andorra case  

                if (Fatalities_hat < X_train[-1]):

                    Fatalities_hat = X_train[-1]                    

            pred_data.append([testdate,country,province,forcast,ConfirmedCases_hat,Fatalities_hat])

    rmsle = RMSLE_Cal(pred_data)

    print(country + ' ' + str(rmsle))
#plot and RMSLE only for Vietnam

plotPrediction(pred_data)
df_past = pd.merge(test,train,on = ['Province/State','Date','Country/Region','Lat','Long'], how = 'inner').drop(['Id','Lat','Long'],axis=1)

df_future = pd.DataFrame(pred_data)

df_future.columns = ['Date','Country/Region','Province/State','ForecastId','ConfirmedCases','Fatalities']

df_future = df_future[(df_future['Date'] > df_past['Date'].max())]



submission = df_past.append(df_future, sort = True)

submission[['ForecastId','ConfirmedCases','Fatalities']].to_csv('submission.csv',index=False)
df=train[train['Country/Region'] == country_plot]

pd.set_option('mode.chained_assignment', None)

df['Cases'] = df['ConfirmedCases'] - df['ConfirmedCases'].shift(1)

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(20, 7))

axes[0].plot(df['Date'], df['ConfirmedCases'], label = 'Total cases ' + df['Date'].max().strftime('%Y-%m-%d'))

axes[0].legend()

axes[1].bar(df['Date'], df['Cases'], label = 'New cases ' + df['Date'].max().strftime('%Y-%m-%d'))

axes[1].legend()

fig.autofmt_xdate()


#from catboost import CatBoostRegressor

#import xgboost as xgb

#from sklearn.linear_model import LinearRegression

#from sklearn.linear_model import RidgeCV,LassoLars

#from sklearn.preprocessing import PolynomialFeatures



from sklearn.tree import DecisionTreeRegressor





#poly_reg = PolynomialFeatures(degree=1)



df_train = train.copy()

df_test = test.copy()

train_period = 10

feature_Confirmed = ['ConfirmedCases'] + ['ConfirmedCases_B' + str(i) for i in range(1,train_period)]

feature_Fatalities = ['Fatalities'] + ['Fatalities_B' + str(i) for i in range(1,train_period)]



pred_data = []



#model = CatBoostRegressor(loss_function = 'RMSE')

#model = LinearRegression()

#model = RidgeCV()

#model = LassoLars()

model = DecisionTreeRegressor(max_depth=5)

#for country in df_train['Country/Region'].unique():

for country in [country_plot,'Vietnam']:

    for province in df_train[(df_train['Country/Region'] == country)]['Province/State'].unique():

        max_train_date = df_train[(df_train['Country/Region'] == country) & (df_train['Province/State'] == province)]['Date'].max()

        for forcast in df_test[(df_test['Country/Region'] == country) & (df_test['Province/State'] == province)]['ForecastId'].unique():

        #for forcast in [test[(test['Country/Region'] == country) & (test['Province/State'] == province)].max()['ForecastId']]:

            testdate = df_test[(df_test['Country/Region'] == country) & (df_test['Province/State'] == province) & (df_test['ForecastId'] == forcast)]['Date'].max()           

            data = df_train[(df_train['Country/Region'] == country) & (df_train['Province/State'] == province) & (df_train['ConfirmedCases'] > 0) & (df_train['Date']<testdate)][['Date','ConfirmedCases','Fatalities']]

            for i in range(1,train_period):

                data['ConfirmedCases_B' + str(i)] = data['ConfirmedCases'].shift(i)

                data['Fatalities_B' + str(i)] = data['Fatalities'].shift(i)   



            # Remove not change day 

            data = data[(data.ConfirmedCases_B1 != data.ConfirmedCases) & (data.ConfirmedCases_B2 != data.ConfirmedCases)]



            X_train_confirmed = data[feature_Confirmed][train_period-1:-1]

            #X_train_confirmed = poly_reg.fit_transform(X_train_confirmed)

            y_train_confirmed = data['ConfirmedCases'][train_period-1:-1]

            feature_Confirmed = ['ConfirmedCases'] + ['ConfirmedCases_B' + str(i) for i in range(1,train_period)]            



            if (len(X_train_confirmed) < 2):

                ConfirmedCases_hat = 0

            elif (len(X_train_confirmed) < train_period):

                ConfirmedCases_hat = X_train_confirmed['ConfirmedCases'].values[-1:][0]

            else:               

                # Only train one time when have not train data to update                

                model.fit(X_train_confirmed,y_train_confirmed)

                X_pred_confirmed = data[feature_Confirmed][-1:]

                #X_pred_confirmed = poly_reg.fit_transform(X_pred_confirmed)

                ConfirmedCases_hat = model.predict(X_pred_confirmed)[-1]

            

            data = data[(data.Fatalities_B1 != data.Fatalities) & (data.Fatalities_B2 != data.Fatalities)]

            X_train_fatalities = data[feature_Fatalities][train_period-1:-1]

            y_train_fatalities = data['Fatalities'][train_period-1:-1]

            feature_Fatalities = ['Fatalities'] + ['Fatalities_B' + str(i) for i in range(1,train_period)]

            if (len(X_train_fatalities) < 1):

                Fatalities_hat = 0

            elif (len(X_train_fatalities) < train_period):

                Fatalities_hat = X_train_fatalities['Fatalities'].values[-1:][0]

            else:               

                # Only train one time when have not train data to update                

                model.fit(X_train_fatalities,y_train_fatalities)

                X_pred_fatalities = data[feature_Fatalities][-1:]

                Fatalities_hat = model.predict(X_pred_fatalities)[-1]

                

            pred_data.append([testdate,country,province,forcast,ConfirmedCases_hat,Fatalities_hat])    

            if (testdate > max_train_date):

                # Update Train data

                lat = df_test[(df_test['Country/Region'] == country) & (df_test['Province/State'] == province) & (df_test['ForecastId'] == forcast)]['Lat'].max()

                long = df_test[(df_test['Country/Region'] == country) & (df_test['Province/State'] == province) & (df_test['ForecastId'] == forcast)]['Long'].max()

                df_train.loc[len(df_train)] = [int(forcast),province,country,lat,long,testdate,float(ConfirmedCases_hat),float(Fatalities_hat)]

                                                
#plot and RMSLE only for Vietnam

plotPrediction(pred_data)
# Forecast base on num of day feature

def CreateInput(data):

    feature = []

    for day in [1,100,200,500,1000]:

        #Get information in train data

        data['Number day from ' + str(day) + ' case'] = 0

        if (train[(train['Country/Region'] == country) & (train['Province/State'] == province) & (train['ConfirmedCases'] > day)]['Date'].count() > 0):

            fromday = train[(train['Country/Region'] == country) & (train['Province/State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        

        else:

            fromday = pd.Timestamp('2020-12-31')        

        for i in range(0, len(data)):

            if (data['Date'].iloc[i] > fromday):

                day_denta = data['Date'].iloc[i] - fromday

                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 

        feature = feature + ['Number day from ' + str(day) + ' case']

    

    return data[feature]

   

    

from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression

import xgboost as xgb

#df_train = train

#df_test = test

#model = CatBoostRegressor(loss_function = 'RMSE')

#model = LinearRegression()

model = xgb.XGBRegressor()

pred_data = []

#for country in train['Country/Region'].unique():

for country in ['Vietnam']:

    for province in train[(train['Country/Region'] == country)]['Province/State'].unique():

        df_train = train[(train['Country/Region'] == country) & (train['Province/State'] == province)]

        df_test = test[(test['Country/Region'] == country) & (test['Province/State'] == province)]

        X_train = CreateInput(df_train)

        y_train_confirmed = df_train['ConfirmedCases']

        y_train_fatalities = df_train['Fatalities']

        X_pred = CreateInput(df_test)

        # Check not change

        if (y_train_confirmed[-5:].mean() == y_train_confirmed.iloc[-1]):            

            y_hat_confirmed = np.repeat(y_train_confirmed.iloc[-1], len(X_pred))

        else:

            model.fit(X_train,y_train_confirmed)        

            y_hat_confirmed = model.predict(X_pred)



        if (y_train_fatalities[-5:].mean() == y_train_fatalities.iloc[-1]):            

            y_hat_fatalities = np.repeat(y_train_fatalities.iloc[-1], len(X_pred))

        else:

            model.fit(X_train,y_train_fatalities)        

            y_hat_fatalities = model.predict(X_pred)            

        

        for i in range(0,len(y_hat_confirmed)):

            pred_data.append([df_test['Date'].iloc[i],country,province,df_test['ForecastId'].iloc[i],y_hat_confirmed[i],y_hat_fatalities[i]])
country_plot = 'Vietnam'

plotPrediction(pred_data)
import tensorflow as tf



train_period = 6

feature_Confirmed = ['ConfirmedCases'] + ['ConfirmedCases_B' + str(i) for i in range(1,train_period)]

feature_Fatalities = ['Fatalities'] + ['Fatalities_B' + str(i) for i in range(1,train_period)]



model = tf.keras.Sequential([

    tf.keras.layers.Dense(256, activation='relu', input_shape=[len(feature_Confirmed)]),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(1)

  ])



optimizer = "rmsprop"

model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])

EPOCHS = 10

BATCH_SIZE = 1



df_train = train.copy()

df_test = test.copy()



pred_data = []

#for country in df_train['Country/Region'].unique():

for country in [country_plot,'Vietnam']:

    for province in df_train[(df_train['Country/Region'] == country)]['Province/State'].unique():

        max_train_date = df_train[(df_train['Country/Region'] == country) & (df_train['Province/State'] == province)]['Date'].max()

        for forcast in df_test[(df_test['Country/Region'] == country) & (df_test['Province/State'] == province)]['ForecastId'].unique():

        #for forcast in [test[(test['Country/Region'] == country) & (test['Province/State'] == province)].max()['ForecastId']]:

            testdate = df_test[(df_test['Country/Region'] == country) & (df_test['Province/State'] == province) & (df_test['ForecastId'] == forcast)]['Date'].max()           

            data = df_train[(df_train['Country/Region'] == country) & (df_train['Province/State'] == province) & (df_train['ConfirmedCases'] > 0) & (df_train['Date']<testdate)][['Date','ConfirmedCases','Fatalities']]

            for i in range(1,train_period):

                data['ConfirmedCases_B' + str(i)] = data['ConfirmedCases'].shift(i)

                data['Fatalities_B' + str(i)] = data['Fatalities'].shift(i)   



            # Remove not change day 

            data = data[(data.ConfirmedCases_B1 != data.ConfirmedCases) & (data.ConfirmedCases_B2 != data.ConfirmedCases)]



            

            X_train_confirmed = data[feature_Confirmed][train_period-1:-1]

            #X_train_confirmed = poly_reg.fit_transform(X_train_confirmed)

            y_train_confirmed = data['ConfirmedCases'][train_period-1:-1]

            

            #model = CatBoostRegressor(loss_function = 'RMSE')

            

            #model = RidgeCV()

            if (len(X_train_confirmed) < 1):

                ConfirmedCases_hat = 0

            elif (len(X_train_confirmed) < train_period):

                ConfirmedCases_hat = X_train_confirmed['ConfirmedCases'].values[-1:][0]

            else:               

                # Only train one time when have not train data to update                

                model.fit(X_train_confirmed, y_train_confirmed ,batch_size=BATCH_SIZE, epochs=EPOCHS, verbose = 0)

                X_pred_confirmed = data[feature_Confirmed][-1:]

                #X_pred_confirmed = poly_reg.fit_transform(X_pred_confirmed)

                ConfirmedCases_hat = model.predict(X_pred_confirmed)[-1][0]

            

            data = data[(data.Fatalities_B1 != data.Fatalities) & (data.Fatalities_B2 != data.Fatalities)]

            

            X_train_fatalities = data[feature_Fatalities][train_period-1:-1]

            y_train_fatalities = data['Fatalities'][train_period-1:-1]

            

            if (len(X_train_fatalities) < 1):

                Fatalities_hat = 0

            elif (len(X_train_fatalities) < train_period):

                Fatalities_hat = X_train_fatalities['Fatalities'].values[-1:][0]

            else:               

                # Only train one time when have not train data to update                

                #model.fit(X_train_fatalities,y_train_fatalities)

                model.fit(X_train_fatalities, y_train_fatalities ,batch_size=BATCH_SIZE, epochs=EPOCHS, verbose = 0)

                X_pred_fatalities = data[feature_Fatalities][-1:]

                Fatalities_hat = model.predict(X_pred_fatalities)[-1][0]

                

            pred_data.append([testdate,country,province,forcast,ConfirmedCases_hat,Fatalities_hat])    

            if (testdate > max_train_date):

                # Update Train data

                lat = df_test[(df_test['Country/Region'] == country) & (df_test['Province/State'] == province) & (df_test['ForecastId'] == forcast)]['Lat'].max()

                long = df_test[(df_test['Country/Region'] == country) & (df_test['Province/State'] == province) & (df_test['ForecastId'] == forcast)]['Long'].max()

                df_train.loc[len(df_train)] = [int(forcast),province,country,lat,long,testdate,float(ConfirmedCases_hat),float(Fatalities_hat)]                                                
#plot and RMSLE only for Vietnam

plotPrediction(pred_data)
# Forecast base on num of day feature each country

feature_day = [1,20,50,100,200,500,1000]

def CreateInput(data):

    feature = []

    for day in feature_day:

        #Get information in train data

        data['Number day from ' + str(day) + ' case'] = 0

        if (train[(train['Country/Region'] == country) & (train['Province/State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):

            fromday = train[(train['Country/Region'] == country) & (train['Province/State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        

        else:

            fromday = pd.Timestamp('2020-12-31')        

        for i in range(0, len(data)):

            if (data['Date'].iloc[i] > fromday):

                day_denta = data['Date'].iloc[i] - fromday

                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 

        feature = feature + ['Number day from ' + str(day) + ' case']

    

    return data[feature]

   

import tensorflow as tf



model = tf.keras.Sequential([

    tf.keras.layers.Dense(64, activation='elu', input_shape=[len(feature_day)]),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(32, activation='elu'),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(1)

  ])



optimizer = "rmsprop"

model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])

EPOCHS = 10

BATCH_SIZE = 1



pred_data = []

#for country in train['Country/Region'].unique():

for country in ['Vietnam']:

    for province in train[(train['Country/Region'] == country)]['Province/State'].unique():

        df_train = train[(train['Country/Region'] == country) & (train['Province/State'] == province)]

        df_test = test[(test['Country/Region'] == country) & (test['Province/State'] == province)]

        X_train = CreateInput(df_train)

        y_train_confirmed = df_train['ConfirmedCases']

        y_train_fatalities = df_train['Fatalities']

        X_pred = CreateInput(df_test)

        # Check not change

        if (y_train_confirmed[-5:].mean() == y_train_confirmed.iloc[-1]):            

            y_hat_confirmed = np.repeat(y_train_confirmed.iloc[-1], len(X_pred))

        else:

            model.fit(X_train,y_train_confirmed,batch_size=BATCH_SIZE, epochs=EPOCHS, verbose = 0)

            y_hat = model.predict(X_pred)

            y_hat_confirmed = [item[0] for item in y_hat]            



        if (y_train_fatalities[-5:].mean() == y_train_fatalities.iloc[-1]):            

            y_hat_fatalities = np.repeat(y_train_fatalities.iloc[-1], len(X_pred))

        else:

            model.fit(X_train,y_train_fatalities,batch_size=BATCH_SIZE, epochs=EPOCHS, verbose = 0)

            y_hat = model.predict(X_pred) 

            y_hat_fatalities = [item[0] for item in y_hat]            

        

        for i in range(0,len(y_hat_confirmed)):

            pred_data.append([df_test['Date'].iloc[i],country,province,df_test['ForecastId'].iloc[i],y_hat_confirmed[i],y_hat_fatalities[i]])
#plot and RMSLE only for Vietnam

plotPrediction(pred_data)
# Forecast base on num of day feature all country

def CreateInput(data):

    feature_day = [5,20,50,100,200,500,1000]

    feature_name = ['Lat','Long']

    #feature_day = [1,20]

    for day in feature_day:

        #Get information in train data

        data['Number day from ' + str(day) + ' case'] = 0

        for country in data['Country/Region'].unique():

            for province in data[data['Country/Region'] == country]['Province/State'].unique():

                fromday = train[(train['Country/Region'] == country) & (train['Province/State'] == province) & (train['ConfirmedCases'] > day)]['Date'].min()

                if (pd.isnull(fromday) == False):

                    data.loc[(data['Country/Region'] == country) & (data['Province/State'] == province),'Number day from ' + str(day) + ' case'] = (data[(data['Country/Region'] == country) & (data['Province/State'] == province)]['Date'] - fromday).dt.days

                #for date in data[(data['Country/Region'] == country) & (data['Province/State'] == province)]['Date'].unique():

                    #data.loc[(data['Country/Region'] == country) & (data['Province/State'] == province) & (data['Date'] == date),'Number day from ' + str(day) + ' case'] = (pd.Timestamp(date) - fromday).days

        data.loc[(data['Number day from ' + str(day) + ' case'] < 0),'Number day from ' + str(day) + ' case'] = 0

        feature_name = feature_name + ['Number day from ' + str(day) + ' case']

    return data[feature_name]

    #return data



import tensorflow as tf



X_train = CreateInput(train.copy())

y_train = train[['ConfirmedCases','Fatalities']]

model = tf.keras.Sequential([

    tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(2)

  ])



optimizer = "rmsprop"

model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])

EPOCHS = 100

BATCH_SIZE = 24

#y_train = train[train['ConfirmedCases']>0]['ConfirmedCases']

model.fit(X_train,y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,verbose = 0)

X_pred = CreateInput(test.copy())

y_hat = model.predict(X_pred)

df_forcast = pd.concat([test[['Date','ForecastId']],pd.DataFrame(y_hat)],axis = 1)

df_forcast.columns = ['Date','ForecastId','ConfirmedCases','Fatalities']

df_forcast.loc[df_forcast['Fatalities'] < 0,'Fatalities'] = 0

df_forcast.loc[df_forcast['ConfirmedCases'] < 0,'ConfirmedCases'] = 0



df_past = pd.merge(test,train,on = ['Province/State','Date','Country/Region','Lat','Long'], how = 'inner').drop(['Id','Lat','Long'],axis=1)

df_future = df_forcast[(df_forcast['Date'] > df_past['Date'].max())]



submission = df_past.append(df_future, sort = True)

#submission[['ForecastId','ConfirmedCases','Fatalities']].to_csv('submission.csv',index=False)

submission[['ForecastId','ConfirmedCases','Fatalities']]
df_val = pd.merge(test,df_forcast[['ForecastId','ConfirmedCases','Fatalities']], how='inner', on = 'ForecastId')

df_val = pd.merge(df_val,train,on=['Date','Country/Region','Province/State','Lat','Long'], how='left')

df_val = df_val.sort_values(by = ['Date','Country/Region'])

df = df_val[df_val['Country/Region'] == 'Vietnam']

#df_val = pd.merge(df,train[['Date','Country/Region','Province/State','ConfirmedCases','Fatalities']],on=['Date','Country/Region','Province/State'], how='left')

val_size = df_val[df_val['ConfirmedCases_y'].notnull() == True].shape[0]

#RMSLE(df_val['ConfirmedCases_x'][0:val_size].values,df_val['ConfirmedCases_y'][0:val_size].values)



fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(20, 7))

axes[0].plot(df['Date'], df['ConfirmedCases_x'], label = 'ConfirmedCase forcast ' + df['Date'].max().strftime('%Y-%m-%d'))

axes[0].plot(df['Date'][0:val_size], df['ConfirmedCases_y'][0:val_size], label = 'ConfirmedCase ' + df[df['ConfirmedCases_y'].isnull() == False]['Date'].max().strftime('%Y-%m-%d'))

axes[0].legend()

axes[1].plot(df['Date'], df['Fatalities_x'], label = 'Fatalities forcast ' + df['Date'].max().strftime('%Y-%m-%d'))

axes[1].plot(df['Date'][0:val_size], df['Fatalities_y'][0:val_size], label = 'Fatalities ' + df[df['Fatalities_y'].isnull() == False]['Date'].max().strftime('%Y-%m-%d'))

axes[1].legend()

fig.autofmt_xdate()
RMSLE(df_val['ConfirmedCases_x'][0:val_size].values,df_val['ConfirmedCases_y'][0:val_size].values)