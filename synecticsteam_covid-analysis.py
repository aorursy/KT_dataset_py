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
import seaborn as sns

import matplotlib.pyplot as plt

from datetime import date

import plotly.express as px

import plotly.io as pio

from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv",parse_dates=['Date'])

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv",parse_dates=['Date'])

submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")
train.info()
train.head()
train.isnull().sum()
train=train.rename(columns={'Country/Region':'Country','Province/State':'State'})

test=test.rename(columns={'Country/Region':'Country','Province/State':'State'})

train['State']=train['State'].fillna('')

test['State']=test['State'].fillna('')

train.head()
submission.head()
train['ActiveCases']=train['ConfirmedCases']-train['Fatalities']
df1=train.groupby(

    [pd.to_datetime(train.Date).dt.strftime('%b %Y'), 'Country']

)['ConfirmedCases'].sum().reset_index(name='TotalCases')

#df1['Month']=pd.DatetimeIndex(df1['Date']).month

#df1['Month-str'] = pd.to_datetime(train.Date).dt.strftime('%b')

df1['Month'] =  pd.to_datetime(df1.Date).dt.strftime('%B')

df1.sort_values('TotalCases', inplace=True)

df1
fig, ax = plt.subplots()

ax.set_xlabel('Month')

ax.set_ylabel('TotalCases')

plt.bar(df1['Month'],df1['TotalCases'])

#plt.xticks(rotation=90)

plt.show()
grouped_data=train.groupby('Date')['Date','ConfirmedCases','Fatalities'].sum().reset_index()

grouped_data.head()

figure1=px.line(grouped_data, x="Date",y="ConfirmedCases",title="Total Confirmed Cases")

figure1.show()



figure2=px.line(grouped_data, x="Date",y="ConfirmedCases", title="Total Confirmed Cases(log value)", log_y=True)

figure2.show()
us_data=train[train['Country']=='US'].reset_index()

us_date=us_data.groupby('Date')['Date','ConfirmedCases','Fatalities'].sum().reset_index()

figure3=px.line(us_date,x='Date',y="ConfirmedCases",title="Total Cases in USA")

figure3.show()
China_data=train[train['Country']=='China'].reset_index()

china_data1=China_data.groupby('Date')['Date','ConfirmedCases','Fatalities'].sum().reset_index()

#china_data1=china_data1[china_data1['Date']<'2020-02-01']

#china_data1

figure4=px.line(china_data1,x='Date',y="ConfirmedCases",title="Total Cases in China")

figure4.show()
Italy_data=train[train['Country']=='Italy'].reset_index()

Italy_data1=Italy_data.groupby('Date')['Date','ConfirmedCases','Fatalities'].sum().reset_index()

figure5=px.line(Italy_data1,x='Date',y="ConfirmedCases",title="Total Cases in Italy")

figure5.show()
Spain_data=train[train['Country']=='Spain'].reset_index()

Spain_data1=Spain_data.groupby('Date')['Date','ConfirmedCases','Fatalities'].sum().reset_index()

figure5=px.line(Spain_data1,x='Date',y="ConfirmedCases",title="Total Cases in Spain")

figure5.show()
India_data=train[train['Country']=='India'].reset_index()

India_data1=India_data.groupby('Date')['Date','ConfirmedCases','Fatalities'].sum().reset_index()

figure6=px.line(India_data1,x='Date',y="ConfirmedCases",title="Total Cases in India")

figure6.show()
country_wise_data = train[train['Date']==max(train['Date'])].reset_index(drop=True).drop('Date', axis=1)

country_wise_data=country_wise_data.groupby('Country')['ConfirmedCases','Fatalities','ActiveCases'].sum().reset_index()

country_wise_data.head()
fig_7 = px.bar(country_wise_data.sort_values('ConfirmedCases').tail(10), x="ConfirmedCases", y="Country", orientation='h', color_discrete_sequence = ['#f38181'],text ='ConfirmedCases')

fig_7.update_layout(title_text="Top 10 countries with the Most Confirmed Cases")

fig_7.show()

fig_8 = px.bar(country_wise_data.sort_values('Fatalities').tail(10), x="Fatalities", y="Country", text='Fatalities', orientation='h', color_discrete_sequence = ['#333333'])

fig_8.update_layout(title_text="Top 10 countries with the Most Fatalities")

fig_8.show()

fig_9 = px.bar(country_wise_data.sort_values('ActiveCases').tail(10), x="ActiveCases", y="Country", text='ActiveCases', orientation='h', color_discrete_sequence = ['#c61951'])

fig_9.update_layout(title_text="Top 10 countries with the Active Cases")

fig_9.show()
#plt.figure(figsize=(15,10))

figure = px.choropleth(train, locations="Country", 

                     color="ConfirmedCases", 

                    hover_name="Country", color_continuous_scale="RdBu",

                    locationmode='country names',range_color=(0, 1000), 

                    title='Total Cases in the world')

figure.show()
china_data=train[train['Country']=='China'].reset_index()

china_df=china_data.groupby('Date')['ConfirmedCases'].sum().reset_index(name='TotalCases')

log_data=china_df['TotalCases']

log_data=log_data.reset_index(drop=False)

log_data.columns=['Timesteps','TotalCases']

log_data
##defining the function to be used

def my_logistic(t,a,b,c):

    return c/(1+a*np.exp(-b*t))

##Randomly initializing a,b,c and setting the bounds

p0=np.random.exponential(size=3)

bounds=(0,[100000.,3.,1000000000.])
import scipy.optimize as optim

x=np.array(log_data['Timesteps'])+1

y=np.array(log_data['TotalCases'])

(a,b,c),cov=optim.curve_fit(my_logistic,x,y,bounds=bounds,p0=p0)

a,b,c
def my_logistic(t):

    return c/(1+a*np.exp(-b*t))


plt.scatter(x,y)

plt.plot(x,my_logistic(x))

plt.title("Logistic model vs the actual trend China")

plt.legend(['Logistic Model','Actual Trend'])

plt.xlabel("Time")

plt.ylabel("Infections")
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

x_features=train

x_features=x_features.drop(columns=['State','ConfirmedCases','Fatalities','Id','Lat','Long','ActiveCases'])

x_features['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

#x_features.loc[:, 'Date'] = x_features.Date.dt.strftime("%m%d")

x_features["Date"]  = x_features["Date"].astype(int)

x_features.Country = le.fit_transform(x_features.Country)

test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)

x_features.head()

x_features.info()
y_target_con = train

y_target_con = y_target_con.drop(columns=['Id','Date','Country','State','Fatalities','Lat','Long','ActiveCases'])

y_target_con.info()

y_target_con.head()
test.head()
test_features = test

test.head()

test_features.Country= le.fit_transform(test_features.Country)

test_features.Date = pd.to_datetime(test_features.Date)

#test_features.loc[:, 'Date'] = test_features.Date.dt.strftime("%m%d")

test_features["Date"]  = test_features["Date"].astype(int)

test_features=test_features.drop(columns=['Long', 'Lat'],axis=1)

test_features.info()

test_features.head()

test_features = test_features.drop(columns=['ForecastId','State'],axis=1)

test_features.info()

test_features.head()

from xgboost import XGBRegressor

model_con1 = XGBRegressor()

con_target = train.ConfirmedCases

model_con1.fit(x_features,con_target)
predict_con= model_con1.predict(test_features)

predict_con
fatalities = train.Fatalities

train.info()

submission.info()

fatalities
model_fat1 = XGBRegressor()

model_fat1.fit(x_features,fatalities)

predict_fat = model_fat1.predict(test_features)

predict_fat
submission = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')

df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

df = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases':predict_con , 'Fatalities': predict_fat})

df_out = pd.concat([df_out, df], axis=0)

df_out.ForecastId = df_out.ForecastId.astype('int')

df_out.to_csv('submission.csv', index=False)