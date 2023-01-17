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
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import warnings

warnings.filterwarnings('ignore')
ca_train = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv",parse_dates=[5],index_col=0)

ca_test = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv",parse_dates=[5],index_col=0)

ca_submission = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv",parse_dates=[5])
ca_train.info()

#ca_test.info()
#any null values?

ca_train.isnull().sum()
#remove unnecessary columns

ca_train.nunique()
for i in range(ca_train.shape[1]):

    if (ca_train.iloc[:,i].nunique() == 1):

        print(ca_train.columns[i],'\t',ca_train.iloc[:,i].value_counts())    
#necessary data

ca_filtered_train = ca_train.copy()

ca_filtered_train.tail(10)
ca_filtered_train['weekday'] = ca_filtered_train.Date.dt.day_name()

ca_filtered_train['Total_affected_people'] = ca_filtered_train['ConfirmedCases']+ca_filtered_train['Fatalities']

ca_filtered_train.tail()
print('Total CA population is 39.56M as of year 2018')

print('confirmed cases population percentage',round((sum(ca_train['ConfirmedCases'])/39.56e6)*100,2),'%')

print('fatalities population percentage',round((sum(ca_train['Fatalities'])/39.56e6)*100,4),'%')
affected_people = pd.DataFrame(ca_filtered_train[47:].groupby('ConfirmedCases')['Fatalities'].sum())

affected_people['Cumulative_deaths_percentage'] = round(affected_people['Fatalities']/sum(affected_people['Fatalities'])*100,2)

sns.scatterplot(x=affected_people.index,y=affected_people.Cumulative_deaths_percentage)
cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

ca_filtered_train.groupby('weekday')['ConfirmedCases','Fatalities'].sum().reindex(cats).plot(kind='bar')

plt.title('daywise analysis')
plt.figure(figsize=(20,12))

ax = sns.lineplot(x='Date',y='ConfirmedCases',data=ca_filtered_train[40:],label='Confirmed_cases')

ax = sns.lineplot(x='Date',y='Fatalities',data=ca_filtered_train[40:],label='Deaths')

ax = sns.lineplot(x='Date',y='Total_affected_people',data=ca_filtered_train[47:],label='Total_affected_people')

plt.xticks(ca_filtered_train['Date'][40:],rotation='vertical')

ax.annotate('Lockdown', xy=('2020-03-20',1177), xytext=('2020-03-17', 1300),arrowprops=dict(facecolor='black',shrink=0.05),fontsize=15)

ax.annotate('Declared State of Emergency', xy=('2020-03-04',0), xytext=('2020-03-04', 500),arrowprops=dict(facecolor='black',shrink=0.05),fontsize=15)

ax.annotate('Increased health care capacity', xy=('2020-03-21',1364), xytext=('2020-03-14', 1600),arrowprops=dict(facecolor='black',shrink=0.05),fontsize=15)

ax.legend(loc='upper left',fontsize='x-large',fancybox=True,shadow=True,borderpad=1)

plt.ylabel('Confirmed_cases and Fatalities')

plt.xticks(rotation='vertical')

plt.title('Trend over a MARCH month',fontsize=20)
ca_train.index = ca_train['Date']

ca_train.drop('Date',axis=1,inplace=True)
ca_train = ca_train[['ConfirmedCases','Fatalities']]

train_confirmedcases = ca_train[['ConfirmedCases']]

train_confirmedcases= train_confirmedcases.iloc[47:]
train_fatalities = ca_train[['Fatalities']]

train_fatalities= train_fatalities.iloc[47:]
#ts_diff_1 = train_confirmedcases - train_confirmedcases.shift()

#ts_diff_1.dropna(inplace=True)
from statsmodels.tsa.arima_model import ARIMA

from pandas import datetime



#fit model on confirmedcases

model = ARIMA(train_confirmedcases, order=(1,1,1)) # (ARIMA) = (1,1,0)

model_fit = model.fit(disp=0)
# predict

start_index = datetime(2020, 3, 12)

end_index = datetime(2020, 4, 23)

forecast_confirmedcases = model_fit.predict(start=start_index, end=end_index)
#fit model on fatalities

model_F = ARIMA(train_fatalities, order=(1,1,0)) # (ARIMA) = (1,1,0)

model_fit_F = model_F.fit(disp=0)
# predict

start_index = datetime(2020, 3, 12)

end_index = datetime(2020, 4, 23)

forecast_fatalities = model_fit_F.predict(start=start_index, end=end_index)
df=pd.concat([forecast_confirmedcases.astype(int),forecast_fatalities.astype(int)],axis=1)
ca_submission.head()
ca_submission['ConfirmedCases'] = list(df[0])

ca_submission['Fatalities'] = list(df[1])

ca_submission.head()
ca_submission = ca_submission[['ForecastId','ConfirmedCases','Fatalities']]

ca_submission.head()
# visualization

plt.figure(figsize=(22,10))

plt.plot(train_confirmedcases.index,train_confirmedcases.ConfirmedCases,label = "original")

plt.plot(forecast_confirmedcases,label = "predicted")

plt.legend(loc='upper left',fontsize='x-large',fancybox=True,shadow=True,borderpad=1)

plt.title('For ConfirmedCases')

plt.show()
# visualization

plt.figure(figsize=(22,10))

plt.plot(train_fatalities.index,train_fatalities.Fatalities,label = "original")

plt.plot(forecast_fatalities,label = "predicted")

plt.legend(loc='upper left',fontsize='x-large',fancybox=True,shadow=True,borderpad=1)

plt.title('For Fatalities')

plt.show()
ca_submission.to_csv('submission.csv',index=False)

ca_submission.head()