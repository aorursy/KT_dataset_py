



#import os

#import numpy as np

import pandas as pd

from fbprophet import Prophet

import copy

#from sklearn.metrics import mean_absolute_error



#from datetime import datetime

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 100) 



train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')





train['Province_State'].fillna('NoState', inplace=True)

train['Location'] = train['Country_Region'].str.cat(train['Province_State'],sep="_")

test['Province_State'].fillna('NoState', inplace=True)

test['Location'] = test['Country_Region'].str.cat(test['Province_State'],sep="_")

len(test['Location'].unique())

test_d=test[test["Date"] > max(train['Date'])]



location=train['Location'].unique()

print(len(location))  



appended_data3 =pd.DataFrame()

for location in location:

    all_df = train[train['Location']==location].groupby('Date')['ConfirmedCases'].sum().reset_index()

    df_prophet = all_df.loc[:,["Date", 'ConfirmedCases']]

    df_prophet.columns = ['ds','y']

    m_d = Prophet(  yearly_seasonality= True,

                    weekly_seasonality = True,

                    daily_seasonality = True,

                    seasonality_mode = 'additive')

    m_d.fit(df_prophet)

    future_d = m_d.make_future_dataframe(periods=35)

    fcst_daily = m_d.predict(future_d)

    fcst_daily['Loca']=location

    data=fcst_daily[['ds','Loca', 'yhat']]

    appended_data3=appended_data3.append(data,ignore_index=True)



appended_data3.rename(columns={'yhat':'ConfirmedCases'}, inplace=True)

appended_data3.rename(columns={'Loca':'Location'}, inplace=True)

appended_data3['Location'].str.strip()

appended_data3['Date']=appended_data3['ds'].dt.strftime('%Y-%m-%d')

sub3=test.merge(appended_data3,  on=('Date','Location'),how='left').reset_index(drop=True)



appended_data3.head





#absolute_errors = []

#MSE_data=appended_data3.merge(train,  on=('Date','Location'),how='inner')

#MSE_data.columns

#MSE_data['abs_diff'] = (MSE_data['ConfirmedCases_y'] - MSE_data['ConfirmedCases_x']).abs()



#absolute_errors += list(MSE_data['abs_diff'].values)

#N = len(absolute_errors)

#mean_absolute_error = sum(absolute_errors)/N





#print("MAE for confirmedcases is", mean_absolute_error )







tr_data=copy.deepcopy(train)

locat=tr_data['Location'].unique()

print(len(locat)) 



appended_data4 =pd.DataFrame()

for location in locat:

    all_df = tr_data[tr_data['Location']==location].groupby('Date')['Fatalities'].sum().reset_index()

    df_prophet = all_df.loc[:,['Date', 'Fatalities']]

    df_prophet.columns = ['ds','y']

    m_d = Prophet( interval_width=0.95, yearly_seasonality= True,

                    weekly_seasonality = True,

                    daily_seasonality = True,

                    seasonality_mode = 'additive')

    m_d.fit(df_prophet)

    future_d = m_d.make_future_dataframe(periods=35)

    fcst_daily = m_d.predict(future_d)

    fcst_daily['Loca']=location

    data=fcst_daily[['ds','Loca', 'yhat']]

    appended_data4=appended_data4.append(data,ignore_index=True)









#absolute_errors2 = []

#MSE_data2=appended_data4.merge(train,  on=('Date','Location'),how='inner')

#MSE_data2.columns

#MSE_data2['abs_diff'] = (MSE_data2['Fatalities_y'] - MSE_data2['Fatalities_x']).abs()



#absolute_errors2 += list(MSE_data2['abs_diff'].values)

#N = len(absolute_errors2)

#mean_absolute_error2 = sum(absolute_errors2)/N





#print("MAE for Fatalities is", mean_absolute_error2 )







appended_data4.rename(columns={'yhat':'Fatalities'}, inplace=True)

appended_data4.rename(columns={'Loca':'Location'}, inplace=True)

appended_data4['Location'].str.strip()

appended_data4['Date']=appended_data4['ds'].dt.strftime('%Y-%m-%d')

sub4=pd.merge(sub3,appended_data4,  on=('Date','Location'),how='left')

sub4.columns









test_d=test[test["Date"] > max(train['Date'])]





com_data=pd.merge(test,train,  on=('Date','Location'),how='inner')



com_data.rename(columns={'Province_State_x':'Province_State'}, inplace=True)

com_data.rename(columns={'Country_Region_x':'Country_Region'}, inplace=True)

com_data.rename(columns={'ConfirmedCases':'confirmed'}, inplace=True)

com_data.rename(columns={'Fatalities':'death'}, inplace=True)

com_data.drop(['Province_State_y', 'Country_Region_y','Location'], axis=1)

after=com_data.drop(columns=['Province_State_y', 'Country_Region_y','Location','Location','Date','Id'])

sub5=pd.merge(sub4,after,  on=('ForecastId'),how='left')







# 2020-04-02 to 2020-04-09 overlap 

sub6=sub5[sub5['Date']>'2020-04-09']

sub6=sub6[['ForecastId', 'ConfirmedCases', 'Fatalities']]

sub7=sub5[(sub5['Date']>'2020-04-01') & (sub5['Date']<'2020-04-10')]







sub8=sub7[['ForecastId', 'confirmed', 'death']].reset_index(drop=True)

sub8.rename(columns={'confirmed':'ConfirmedCases'}, inplace=True)

sub8.rename(columns={'death':'Fatalities'}, inplace=True)



subm=sub6.append(sub8,ignore_index=True)

subb=submission['ForecastId']

subm2=pd.merge(subb,subm,on=('ForecastId'),how='left')

subm2.columns

tt2=pd.DataFrame()



tt2['ConfirmedCases']=subm2['ConfirmedCases']

tt2['Fatalities']=subm2['Fatalities']

tt2[tt2['Fatalities'] < 0] = 0

tt2[tt2['ConfirmedCases'] < 0] = 0

tt2['ForecastId']=subm2['ForecastId']

tt2.reset_index(drop=True, inplace=True)

tt2.set_index('ForecastId')

tt2['Fatalities'].fillna('0', inplace=True)

tt2['ConfirmedCases'].fillna('0', inplace=True)









tt2.to_csv("submission.csv", index=False, columns=["ForecastId", "ConfirmedCases", "Fatalities"])