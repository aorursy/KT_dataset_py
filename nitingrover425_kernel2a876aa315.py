import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import datetime as dt

import folium
!kaggle kaggle competitions download -c covid19-global-forecasting-week-1
!pip install kaggle
!mkdir .kaggle
import json

token = {"username":'nitingrover425','key':'c22685e02df7d46edd199e441980c448'}

with open('/content/.kaggle/kaggle.json', 'w') as file:

    json.dump(token, file)
!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
!kaggle config set -n path -v{/content}
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets list
!kaggle competitions download -c covid19-global-forecasting-week-1 -p /content
train_data = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
train_data.head(10)
train_data.tail(10)
train_data.shape
train_data.isnull().sum()
train_data = train_data.rename(columns = {"Province/State":"State" , "Country/Region":"Country" } )
train_data.columns
train_data = train_data.fillna("Not Available")

train_data.isnull().sum()
train_data.head(10)
train_data[train_data['ConfirmedCases']<0]
train_data[train_data['Fatalities']<0]
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data[train_data['ConfirmedCases'] == train_data['ConfirmedCases'].max()]
train_data['ConfirmedCases'].max()
train_data['Date'].max()
date_country_total = train_data.groupby(['Date']).sum()

date_country_total.head(10)
date_country_total['Total Cases'] = date_country_total['ConfirmedCases'].cumsum() 
date_country_total.tail()
date_country_total.reset_index(inplace = True)

date_country_total.columns
px.line(date_country_total,x = 'Date', y = 'ConfirmedCases' , title = 'COVID-19 Cases:World')
px.line(date_country_total.tail() , x = 'Date' , y = 'ConfirmedCases' , title = 'COVID-19 Cases:World[Past 5 days]')
country_cases = train_data.groupby(['Country' , 'Lat' , 'Long']).sum()

country_cases.reset_index(inplace=True)

country_cases
country_cases.ConfirmedCases.dtype

country_cases.reset_index(inplace=True)

country_cases.head()
folium_map = folium.Map(location=[20.5936832, 78.962883],

                            zoom_start=1.5,

                            tiles="CartoDB dark_matter"

                            )



for index,row in country_cases.iterrows():

  radius_confirmed = row['ConfirmedCases']/1000



  color = '#ffd700' #gold #confirmed

  #radius = radius_confirmed



  radius_fatal = row['Fatalities']/1000



  color_f = '#ff0000'  #red #death

  #radius = radius_fatal



  folium.CircleMarker(location=(row['Lat'],row['Long']),radius=radius_confirmed,color=color,fill=True ).add_to(folium_map)

  folium.CircleMarker(location=(row['Lat'],row['Long']),radius=radius_fatal,color=color_f,fill=True ).add_to(folium_map)



  

  
folium_map
columns = train_data.columns.tolist()

columns

train_data['Date'] = pd.to_numeric(train_data['Date'])
columns = [c for c in columns if c not in['Id','ConfirmedCases','Fatalities','State','Country']]

X_train = train_data[columns]

Y1_train = train_data['ConfirmedCases']

Y2_train = train_data['Fatalities']

print(X_train.shape)

print(Y1_train.shape)

print(Y2_train.shape)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100 ,random_state=0)

regressor.fit(X_train,Y1_train)

#regressor.fit(X_train,Y2_train)
test_data = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
test_data.head()
columnst = test_data.columns.tolist()

columnst

test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data['Date'] = pd.to_numeric(test_data['Date'])
columnst = [c for c in columnst if c not in ['ForecastId', 'Province/State', 'Country/Region']]

columnst
y1_pred = regressor.predict(test_data[columnst])
regressor.fit(X_train,Y2_train)

y2_pred = regressor.predict(test_data[columnst])
pred1 = pd.DataFrame(y1_pred)

pred2 = pd.DataFrame(y2_pred)

sub_df = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')

sub_df.head()
datasets = pd.concat([sub_df['ForecastId'],pred1,pred2],axis=1)

datasets.columns = ['ForecastId','ConfirmedCases','Fatalities']

datasets.to_csv('submission.csv',index=False)