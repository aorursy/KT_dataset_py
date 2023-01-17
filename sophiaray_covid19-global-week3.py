



#import os

#import numpy as np

import pandas as pd

from fbprophet import Prophet

#from datetime import datetime

train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

train['Province_State'].fillna('NoState', inplace=True)

train['Location'] = train['Country_Region'].str.cat(train['Province_State'],sep="_")

test['Province_State'].fillna('NoState', inplace=True)

test['Location'] = test['Country_Region'].str.cat(test['Province_State'],sep="_")

len(test['Location'].unique())

test_d=test[test["Date"] > max(train['Date'])]

print("predict for", test_d['Date'].nunique() ,"days")

print("prediction Dates go from day", min(test_d['Date']), "to day", max(test_d['Date']), ", a total of", test_d['Date'].nunique(), "days")

max(train['Date'])

print("Number of Country_Region: ", train['Country_Region'].nunique()) #163 countries

print("Dates go from day", min(test['Date']), "to day", max(test['Date']), ", a total of", test['Date'].nunique(), "days")

print("Dates go from day", min(train['Date']), "to day", max(train['Date']), ", a total of", train['Date'].nunique(), "days")

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

print("prediction Dates go from day", min(test['Date']), "to day", max(test['Date']), ", a total of", test['Date'].nunique(), "days")

print("prediction Dates go from day", min(appended_data3['Date']), "to day", max(appended_data3['Date']), ", a total of", appended_data3['Date'].nunique(), "days")

sub3=test.merge(appended_data3,  on=('Date','Location'),how='left').reset_index(drop=True)



location



tr_data=train[train['Location'] !='Zambia_NoState' ]

tr_data=tr_data[tr_data['Location']!= 'Libya_NoState']

tr_data=tr_data[tr_data['Location']!= "Congo (Brazzaville)_NoState" ]

locat=tr_data['Location'].unique()

print(len(locat)) 



appended_data4 =pd.DataFrame()

for location in locat:

    all_df = tr_data[tr_data['Location']==location].groupby('Date')['Fatalities'].sum().reset_index()

    df_prophet = all_df.loc[:,['Date', 'Fatalities']]

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

    appended_data4=appended_data4.append(data,ignore_index=True)



appended_data4.rename(columns={'yhat':'Fatalities'}, inplace=True)

appended_data4.rename(columns={'Loca':'Location'}, inplace=True)

appended_data4['Location'].str.strip()

appended_data4['Date']=appended_data4['ds'].dt.strftime('%Y-%m-%d')

sub4=pd.merge(sub3,appended_data4,  on=('Date','Location'),how='left')

train_m=train

test_d=test[test["Date"] > max(train['Date'])]





com_data=pd.merge(test,train,  on=('Date','Location'),how='inner')

print("Dates go from day", min(com_data['Date']), "to day", max(com_data['Date']), ", a total of", com_data['Date'].nunique(), "days")

com_data.rename(columns={'Province_State_x':'Province_State'}, inplace=True)

com_data.rename(columns={'Country_Region_x':'Country_Region'}, inplace=True)

com_data.rename(columns={'ConfirmedCases':'confirmed'}, inplace=True)

com_data.rename(columns={'Fatalities':'death'}, inplace=True)

com_data.drop(['Province_State_y', 'Country_Region_y','Location'], axis=1)

after=com_data.drop(columns=['Province_State_y', 'Country_Region_y','Location','Location','Date','Id'])

sub5=pd.merge(sub4,after,  on=('ForecastId'),how='left')

sub6=sub5[sub5['Date']>'2020-04-02']

print("Dates go from day", min(sub6['Date']), "to day", max(sub6['Date']), ", a total of", sub6['Date'].nunique(), "days")

sub6=sub6[['ForecastId', 'ConfirmedCases', 'Fatalities']]

sub7=sub5[sub5['Date']<'2020-04-03']

sub7=sub7[['ForecastId', 'confirmed', 'death']].reset_index(drop=True)

sub7.rename(columns={'confirmed':'ConfirmedCases'}, inplace=True)

sub7.rename(columns={'death':'Fatalities'}, inplace=True)

subm=sub6.append(sub7,ignore_index=True)

subm2=pd.merge(submission,subm,on=('ForecastId'),how='left')



tt2=pd.DataFrame()



tt2['ConfirmedCases']=subm2['ConfirmedCases_y']

tt2['Fatalities']=subm2['Fatalities_y']

tt2[tt2['Fatalities'] < 0] = 0

tt2[tt2['ConfirmedCases'] < 0] = 0

tt2['ForecastId']=subm2['ForecastId']

tt2.reset_index(drop=True, inplace=True)

tt2.set_index('ForecastId')

tt2['Fatalities'].fillna('0', inplace=True)

tt2['ConfirmedCases'].fillna('0', inplace=True)









tt2.to_csv("submission.csv", index=False, columns=["ForecastId", "ConfirmedCases", "Fatalities"])