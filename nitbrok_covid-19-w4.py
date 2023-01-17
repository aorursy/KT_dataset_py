import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

plt.style.use('dark_background')

import warnings

warnings.filterwarnings('ignore')

import seaborn
train_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
print(train_data.shape)

print(test_data.shape)
train_data.rename(columns={'Country_Region':'Country'}, inplace=True)

test_data.rename(columns={'Country_Region':'Country'}, inplace=True)

train_data.rename(columns={'Province_State':'Province'}, inplace=True)

test_data.rename(columns={'Province_State':'Province'}, inplace=True)
india=train_data.loc[train_data['Country']=='India']

plt.figure(figsize=(20,10))

plt.bar(india.Date,india.ConfirmedCases)

plt.bar(india.Date,india.Fatalities)

plt.title('INDIA Circumstances')

plt.xticks(rotation=90)

plt.show()
confirmed=train_data.groupby('Date').sum()['ConfirmedCases'].reset_index()

deaths=train_data.groupby('Date').sum()['Fatalities'].reset_index()

plt.figure(figsize=(22,9))

plt.bar(confirmed['Date'],confirmed['ConfirmedCases'])

plt.title('World Circumstances')

plt.bar(deaths['Date'],deaths['Fatalities'])

plt.xticks(rotation=90)

plt.show()
train_data['Date'] = pd.to_datetime(train_data['Date'])

test_data['Date'] = pd.to_datetime(test_data['Date'])
train_data['Date']=train_data['Date'].dt.strftime('%m%d')

test_data['Date']=test_data['Date'].dt.strftime('%m%d')

train_data['Date']=train_data['Date'].astype(int)

test_data['Date']=test_data['Date'].astype(int)
train_data["Province"]=train_data.apply(lambda row: str(row['Country']) if pd.isnull(row['Province']) else row['Province'] ,axis=1)

test_data["Province"]=test_data.apply(lambda row: str(row['Country']) if pd.isnull(row['Province']) else row['Province'] ,axis=1)
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

train_data['Province']=encoder.fit_transform(train_data['Province'])

test_data['Province']=encoder.fit_transform(test_data['Province'])

train_data['Country']=encoder.fit_transform(train_data['Country'])

test_data['Country']=encoder.fit_transform(test_data['Country'])
train_data.drop(['Id'],axis=1,inplace=True)
train_data.head()
features=['Province','Country','Date']

target_1=['ConfirmedCases']

target_2=['Fatalities']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_data[target_1]=train_data[target_1].astype(int)

train_data[target_2]=train_data[target_2].astype(int)
from xgboost import XGBRegressor

xgb_clf_1=XGBRegressor(n_estimators=1300)

xgb_clf_2=XGBRegressor(n_estimators=1300)
xgb_df=pd.DataFrame({'ForecastId':[],'ConfirmedCases':[],'Fatalities':[]})

for country in train_data['Country'].unique():

    Province=train_data.loc[train_data.Country == country, :].Province.unique().tolist()

    for province in Province:

        new_train=train_data.loc[train_data['Province']==province]

        new_train[features] = scaler.fit_transform(new_train[features].values)



        xgb_clf_1.fit(new_train[features],new_train[target_1])

        xgb_clf_2.fit(new_train[features],new_train[target_2])



        new_test=test_data.loc[test_data['Province']==province]

        new_test_FI=new_test.ForecastId

        new_test[features]=scaler.transform(new_test[features])

        new_pred_1=xgb_clf_1.predict(new_test[features])

        new_pred_2=xgb_clf_2.predict(new_test[features])



        new_df=pd.DataFrame({'ForecastId':new_test_FI,'ConfirmedCases':new_pred_1,'Fatalities':new_pred_2})



        xgb_df=pd.concat([xgb_df,new_df],axis=0)
xgb_df=xgb_df.drop_duplicates()

xgb_df['ForecastId']=xgb_df['ForecastId'].astype(int)

xgb_df['ConfirmedCases']=np.round(xgb_df['ConfirmedCases'])

xgb_df['Fatalities']=np.round(xgb_df['Fatalities'])
xgb_df.shape
xgb_df.to_csv('submission.csv',index=False)