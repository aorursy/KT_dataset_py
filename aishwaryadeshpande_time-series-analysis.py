# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from datetime import date

from datetime import time

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "time_data.csv"):  

    csv = df.to_csv(index= False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
train_data = pd.read_csv("/kaggle/input/train-data/Train_SU63ISt.csv")

test_data = pd.read_csv("/kaggle/input/test-data/Test_0qrQsBZ.csv")



train_original=train_data.copy() 

test_original=test_data.copy()
train_data.head()
train_data['Datetime'] = pd.to_datetime(train_data.Datetime,format='%d-%m-%Y %H:%M') 

test_data['Datetime'] = pd.to_datetime(test_data.Datetime,format='%d-%m-%Y %H:%M') 

test_original['Datetime'] = pd.to_datetime(test_original.Datetime,format='%d-%m-%Y %H:%M') 

train_original['Datetime'] = pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')
for i in (train_data, test_data, test_original, train_original):

    i['year']=i.Datetime.dt.year 

    i['month']=i.Datetime.dt.month 

    i['day']=i.Datetime.dt.day

    i['Hour']=i.Datetime.dt.hour 
train_data['dayOfWeek']=train_data['Datetime'].dt.dayofweek 
def applyer(row):

    if row.dayofweek == 5 or row.dayofweek == 6:

        return 1

    else:

        return 0 

temp2 = train_data['Datetime'].apply(applyer) 

train_data['weekend']=temp2
train_data.index = train_data['Datetime'] # indexing the Datetime to get the time period on the x-axis. 

df=train_data.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis. 

ts = df['Count'] 

plt.figure(figsize=(16,8)) 

plt.plot(ts, label='Passenger Count') 

plt.title('Time Series') 

plt.xlabel("Time(year-month)") 

plt.ylabel("Passenger count") 

plt.legend(loc='best')
train_data.groupby("year")['Count'].mean().plot.bar()
train_data.groupby("month")['Count'].mean().plot.bar()
train_data.groupby("day")['Count'].mean().plot.bar()
train_data.groupby("Hour")['Count'].mean().plot.bar()
train_data.groupby("weekend")['Count'].mean().plot.bar()
Train=train_data.loc['2012-08-25':'2014-06-24'] 

valid=train_data.loc['2014-06-25':'2014-09-25']
def submission(model):

    predict=model.forecast(len(test_data))

    submission = test_data[["ID"]]

    submission["Count"] = predict 

    return submission
from sklearn.metrics import mean_squared_error



df = np.asarray(Train.Count)

y_pred = valid.copy()

y_pred['naive'] = df[len(df)-1]



rmse = np.sqrt(mean_squared_error(valid['Count'],y_pred['naive']))

print(rmse)
plt.figure(figsize=(12,8)) 

plt.plot(Train.index,Train['Count'],label = "Train")

plt.plot(valid.index,valid['Count'],label = "Valid")

plt.plot(y_pred.index,y_pred['naive'],label = "Naive")

plt.legend(loc='best') 

plt.title("Naive Forecast")

plt.show()
y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(5).mean().iloc[-1]



plt.figure(figsize=(12,8)) 

plt.plot(Train.index,Train['Count'],label = "Train")

plt.plot(valid.index,valid['Count'],label = "Valid")

plt.plot(y_hat_avg.index,y_hat_avg['moving_avg_forecast'],label = "average")

plt.legend(loc='best') 

plt.title("Moving Averages")

plt.show()



rmse = np.sqrt(mean_squared_error(valid['Count'],y_hat_avg['moving_avg_forecast']))

print(rmse)
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



model = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.6,optimized=False)



pred1 = valid.copy()

pred1["SES"] = model.forecast(len(valid))



plt.figure(figsize=(12,8)) 

plt.plot(Train.index,Train['Count'],label = "Train")

plt.plot(valid.index,valid['Count'],label = "Valid")

plt.plot(pred1.index,pred1["SES"],label = "smothing")

plt.legend(loc='best') 

plt.title("Moving Averages")

plt.show()



rmse = np.sqrt(mean_squared_error(valid['Count'],pred1['SES']))

print(rmse)
fit2 = Holt(np.asarray(Train['Count'])).fit()

y_holt = valid.copy()

y_holt["Holt"] = fit2.forecast(len(valid))



plt.figure(figsize=(12,8)) 

plt.plot(Train.index,Train['Count'],label = "Train")

plt.plot(valid.index,valid['Count'],label = "Valid")

plt.plot(y_holt.index,y_holt["Holt"],label = "Holt")

plt.legend(loc='best') 

plt.title("Moving Averages")

plt.show()



rmse = np.sqrt(mean_squared_error(valid['Count'],y_holt['Holt']))

print(rmse)
#Submission



df =submission(fit2)

predict=fit2.forecast(len(test_data))

test_data['prediction']=predict



create_download_link(df)
train_original['ratio'] = train_original['Count']/train_original['Count'].sum() 



temp=train_original.groupby(['Hour'])['ratio'].sum()

pd.DataFrame(temp, columns=['Hour','ratio']).to_csv('GROUPby.csv') 



temp2=pd.read_csv("GROUPby.csv") 

temp2=temp2.drop('Hour.1',1) 
# Predicting by merging merge and temp2 

prediction=pd.merge(test_data, temp2, on='Hour', how='left') 



prediction['Count']=prediction['prediction']*prediction['ratio']*24 

submission=prediction.drop(['Datetime', 'year','month','day','prediction','Hour', 'ratio'],axis=1) 

create_download_link(submission)

y_hat_avg = valid.copy() 

expSm = ExponentialSmoothing(np.asarray(Train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit() 

y_hat_avg['Holt_Winter'] = expSm.forecast(len(valid)) 



plt.figure(figsize=(16,8)) 

plt.plot(Train['Count'],label = 'Train')

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')

plt.show()



rms = np.sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_Winter)) 

print(rms)