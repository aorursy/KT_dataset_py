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
#pd.set_option('mode.chained_assignment', None)

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
train[train['Country_Region'] == 'US'].sort_values('ConfirmedCases',ascending = False)
train[train['Country_Region'] == 'US'].groupby(['Date']).sum()
import altair as alt



df_total = train.groupby(['Date']).sum().reset_index()

base = alt.Chart(df_total).encode(alt.X('Date:T'))

line_A = base.mark_line(point=False, color='#5276A7').encode(alt.Y('ConfirmedCases:Q', axis=alt.Axis(titleColor='#5276A7')))

line_B = base.mark_line(point=False, color='#F18727').encode(alt.Y('Fatalities:Q', axis=alt.Axis(titleColor='#F18727')))

alt.layer(line_A, line_B).resolve_scale(y='independent')
alt.Chart(df_total, title = "Total of World").mark_line().encode(

    alt.X('Date:T'),

    alt.Y(alt.repeat("column"), type='quantitative'),

    #color='species:N'

#).properties(

#    width=500,

#    height=200

).repeat(

    #row=['ConfirmedCases', 'Fatalities'],

    column=['ConfirmedCases', 'Fatalities']

)#.interactive()
df_now = train.groupby(['Date','Country_Region']).sum().sort_values(['Country_Region','Date']).reset_index()

df_now['New Cases'] = df_now['ConfirmedCases'].diff()

df_now['New Fatalities'] = df_now['Fatalities'].diff()

df_now = df_now.groupby('Country_Region').apply(lambda group: group.iloc[-1:]).reset_index(drop = True)
alt.Chart(df_now.sort_values('ConfirmedCases', ascending=False).head(15)).mark_bar().encode(

    alt.X(alt.repeat("column"), type='quantitative'),

    alt.Y('Country_Region', type='nominal', sort='-x'),

    tooltip=['Country_Region', 'ConfirmedCases', 'Fatalities','New Cases', 'New Fatalities'],

    color = alt.value('orange')

).repeat(column=['New Cases', 'New Fatalities'])
alt.Chart(df_now.sort_values('ConfirmedCases', ascending=False).head(15), title = "Top country").mark_bar().encode(

    alt.X(alt.repeat("column"), type='quantitative'),

    alt.Y('Country_Region', type='nominal', sort='-x'),

    tooltip=['Country_Region', 'ConfirmedCases', 'Fatalities','New Cases', 'New Fatalities'],

    color = alt.value('orange')

).repeat(column=['ConfirmedCases', 'Fatalities'])
train['Province_State'].fillna('', inplace=True)

test['Province_State'].fillna('', inplace=True)

train['Date'] =  pd.to_datetime(train['Date'])

test['Date'] =  pd.to_datetime(test['Date'])

train = train.sort_values(['Country_Region','Province_State','Date'])

test = test.sort_values(['Country_Region','Province_State','Date'])



# Fix error in train data

train[['ConfirmedCases', 'Fatalities']] = train.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases', 'Fatalities']].transform('cummax') 
from tqdm import tqdm

import warnings



def RMSLE(pred,actual):

    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))



feature_day = [1,20,50,100,200,500,1000,5000,10000,15000,20000,50000,100000,200000, 500000]



def CreateInput(data):

    feature = []

    for day in feature_day:

        #Get information in train data

        data.loc[:,'Number day from ' + str(day) + ' case'] = 0

        if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):

            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        

        else:

            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       

        for i in range(0, len(data)):

            if (data['Date'].iloc[i] > fromday):

                day_denta = data['Date'].iloc[i] - fromday

                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 

        feature = feature + ['Number day from ' + str(day) + ' case']

    

    return data[feature]
from statsmodels.tsa.statespace.sarimax import SARIMAX



pred_data_all = pd.DataFrame()

with tqdm(total=len(train['Country_Region'].unique())) as pbar:

    for country in train['Country_Region'].unique():

    #for country in ['US']:

        for province in train[(train['Country_Region'] == country)]['Province_State'].unique():

        #for province in ['New York']:

            with warnings.catch_warnings():

                warnings.filterwarnings("ignore")

                df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]

                df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                X_train = CreateInput(df_train)

                y_train_confirmed = df_train['ConfirmedCases'].ravel()

                y_train_fatalities = df_train['Fatalities'].ravel()

                X_pred = CreateInput(df_test)



                # Define feature to use by X_pred

                feature_use = X_pred.columns[0]

                for i in range(X_pred.shape[1] - 1,0,-1):

                    if (X_pred.iloc[0,i] > 10):

                        feature_use = X_pred.columns[i]

                        break

                idx = X_train[X_train[feature_use] == 0].shape[0]          

                adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)



                adjusted_y_train_confirmed = y_train_confirmed[idx:]

                adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)

                

                pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()

                min_test_date = pred_data['Date'].min()            



                model = SARIMAX(adjusted_y_train_confirmed, order=(1,1,0),

                                #seasonal_order=(1,1,0,12),

                                measurement_error=True).fit(disp=False)

                y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

                

                y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values

                y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)                        



                model = SARIMAX(adjusted_y_train_fatalities, order=(1,1,0),

                                #seasonal_order=(1,1,0,12),

                                measurement_error=True).fit(disp=False)

                y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

                                

                y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values

                y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)            



                pred_data['ConfirmedCases_hat'] = y_hat_confirmed

                pred_data['Fatalities_hat'] = y_hat_fatalities

                pred_data_all = pred_data_all.append(pred_data)

        pbar.update(1)

    

df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')

df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0

df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0

df_val = df_val.round({'ConfirmedCases': 0, 'Fatalities': 0})

df_val_1 = df_val.copy()
!pip install pmdarima
import pmdarima as pm



pred_data_all = pd.DataFrame()

with tqdm(total=len(train['Country_Region'].unique())) as pbar:

    for country in train['Country_Region'].unique():

    #for country in ['Vietnam']:

        for province in train[(train['Country_Region'] == country)]['Province_State'].unique():

            with warnings.catch_warnings():

                warnings.filterwarnings("ignore")

                df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]

                df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                X_train = CreateInput(df_train)

                y_train_confirmed = df_train['ConfirmedCases'].ravel()

                y_train_fatalities = df_train['Fatalities'].ravel()

                X_pred = CreateInput(df_test)



                # Define feature to use by X_pred

                feature_use = X_pred.columns[0]

                for i in range(X_pred.shape[1] - 1,0,-1):

                    if (X_pred.iloc[0,i] > 10):

                        feature_use = X_pred.columns[i]

                        break

                idx = X_train[X_train[feature_use] == 0].shape[0]          

                adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)

                adjusted_y_train_confirmed = y_train_confirmed[idx:]

                adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)



                pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()

                min_test_date = pred_data['Date'].min()            



                model = pm.auto_arima(adjusted_y_train_confirmed, suppress_warnings=True, seasonal=False, error_action="ignore")            

                y_hat_confirmed = model.predict(pred_data[pred_data['Date'] > max_train_date].shape[0])

                y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values

                y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)                        



                model = pm.auto_arima(adjusted_y_train_fatalities, suppress_warnings=True, seasonal=False, error_action="ignore")            

                y_hat_fatalities = model.predict(pred_data[pred_data['Date'] > max_train_date].shape[0])

                y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values

                y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)            



                pred_data['ConfirmedCases_hat'] = y_hat_confirmed

                pred_data['Fatalities_hat'] = y_hat_fatalities

                pred_data_all = pred_data_all.append(pred_data)

        pbar.update(1)

    

df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')

df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0

df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0



df_val_2 = df_val.copy()
df_SARIMAX = df_val_1.groupby(['Date']).sum()[['ConfirmedCases_hat','Fatalities_hat']].reset_index()

df_SARIMAX['Model'] = 'SARIMAX'

df_ARIMA = df_val_2.groupby(['Date']).sum()[['ConfirmedCases_hat','Fatalities_hat']].reset_index()

df_ARIMA['Model'] = 'ARIMA'



source = pd.concat([df_SARIMAX,df_ARIMA], ignore_index=True)



alt.Chart(source).mark_line(point=True).encode(

    x='Date',

    y=alt.Y(alt.repeat("column"), type='quantitative'),

    tooltip=['Model', 'Model', 'ConfirmedCases_hat', 'Fatalities_hat'],

    color='Model',

    #strokeDash='Country_Region'

).repeat(column=['ConfirmedCases_hat', 'Fatalities_hat'])
df_val = df_val_1

alt.Chart(df_val.groupby(['Date']).sum().reset_index()).mark_line(point=True).encode(

    alt.X('Date:T'), 

    alt.Y(alt.repeat("column"), type='quantitative'),

    color=alt.condition(alt.datum.Date >= alt.expr.toDate(df_now.Date.values[0]), 

                        alt.value('orange'),     

                        alt.value('steelblue')),

    tooltip=['Date', 'ConfirmedCases_hat', 'Fatalities_hat']

).repeat(column=['ConfirmedCases_hat', 'Fatalities_hat'])

source = df_val[df_val['Country_Region'].isin(df_now.sort_values('ConfirmedCases', ascending=False).head(5)['Country_Region'].values)]

source = source.groupby(['Date','Country_Region']).sum().sort_values(['Country_Region','Date']).reset_index()

alt.Chart(source).mark_line(point=True).encode(

    x='Date',

    y=alt.Y(alt.repeat("column"), type='quantitative'),

    tooltip=['Date', 'Country_Region', 'ConfirmedCases_hat', 'Fatalities_hat'],

    color='Country_Region',

    #strokeDash='Country_Region'

).repeat(column=['ConfirmedCases_hat', 'Fatalities_hat'])
country = 'US'

source = df_val[df_val['Country_Region'] == country]

source = source.groupby(['Date','Country_Region']).sum().sort_values(['Country_Region','Date']).reset_index()

alt.Chart(source).mark_line(point=True).encode(

    x='Date',

    y=alt.Y(alt.repeat("column"), type='quantitative'),

    tooltip=['Date', 'ConfirmedCases_hat', 'Fatalities_hat'],

    color=alt.condition(alt.datum.Date >= alt.expr.toDate(df_now.Date.values[0]), 

                        alt.value('orange'),     

                        alt.value('steelblue')),

).repeat(column=['ConfirmedCases_hat', 'Fatalities_hat'])
df_val = df_val_2

submission = df_val[['ForecastId','ConfirmedCases_hat','Fatalities_hat']]

submission.columns = ['ForecastId','ConfirmedCases','Fatalities']

submission = submission.round({'ConfirmedCases': 0, 'Fatalities': 0})

submission.to_csv('submission.csv', index=False)

submission