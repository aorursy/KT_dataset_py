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
import pandas as pd

import numpy as np

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=15,10
data=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

print(data.shape)

data.head()
#Dropping the columns Sno and Last Update

data.drop(['SNo','Last Update'],axis=1,inplace=True)



#Getting additonal Active cases for better implementation and information

data['Active']=data['Confirmed']-data['Deaths']-data['Recovered']



data.tail()
print(data.isna().sum())



data['Province/State']=data['Province/State'].fillna('Not Specified')

print('\n\n',data.isna().sum())
print(data.dtypes)



data['Confirmed']=data['Confirmed'].astype(int)

data['Recovered']=data['Recovered'].astype(int)

data['Deaths']=data['Deaths'].astype(int)

data['Active']=data['Active'].astype(int)



print('\n\n',data.dtypes)
china_cases=data[data['ObservationDate']==max(data['ObservationDate'])].reset_index(drop=True)

china_cases=china_cases.groupby('Country/Region')['Confirmed'].sum()['Mainland China']

china_cases
data_nc=data[data['Country/Region']!='Mainland China']

data_nc
#China cases included

data_per_day=data.groupby('ObservationDate')[['Confirmed','Deaths','Recovered','Active']].sum()

#China cases excluded

data_per_day_nc=data_nc.groupby('ObservationDate')[['Confirmed','Deaths','Recovered','Active']].sum()
data_per_day.plot(kind='line',figsize=(20,8))

plt.ylabel('Number of Cases',size=20)

plt.xlabel('Dates',size=20)

plt.title('Number of cases including China(Initially)',size=20)

plt.legend(prop={'size':'15'})
#Data for Countries except China

data_per_day_nc.plot(kind='line',figsize=(20,8))

plt.ylabel('Number of Cases',size=20)

plt.xlabel('Dates',size=20)

plt.title('Number of cases excluding China(Initially)',size=20)

plt.legend(prop={'size':'15'})
from fbprophet import Prophet



p=Prophet()
p.add_seasonality(name='monthly',period=30.5,fourier_order=5)
print(data_per_day.shape)



cases=data_per_day.reset_index()

cases_nc=data_per_day_nc.reset_index()
confirmed_cases=cases_nc[['ObservationDate','Confirmed']]

recovered_cases=cases_nc[['ObservationDate','Recovered']]

death_cases=cases_nc[['ObservationDate','Deaths']]

active_cases=cases_nc[['ObservationDate','Active']]
confirmed_cases.rename(columns={'ObservationDate':'ds','Confirmed':'y'},inplace=True)
#Fit Model

p.fit(confirmed_cases)
#Future Dates

future_dates=p.make_future_dataframe(periods=30)

future_dates
#Prediction

prediction=p.predict(future_dates)
#Plot Prediction

p.plot(prediction,figsize=(20,8))

plt.xlabel('Dates',size=20)

plt.ylabel('Number of Confirmed cases',size=20)

plt.title('Predicted Number of Confirmed Cases',size=20)
p.plot_components(prediction)
#Find Points/Dates for change

from fbprophet.plot import add_changepoints_to_plot

fig=p.plot(prediction)

c=add_changepoints_to_plot(fig.gca(),p,prediction)
prediction.tail().T
prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
k=len(prediction)

for i in range(confirmed_cases.shape[0],k) :

  print('Prediction of Confirmed cases for',prediction['ds'][i],'is ',round(prediction['yhat'][i].astype(int))+china_cases)
data1=data[data['ObservationDate'] == max(data['ObservationDate'])].reset_index(drop=True)

df = data1.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

df1 = df.sort_values(by='Confirmed', ascending=False).reset_index(drop=True)

df1 = df1[['Country/Region', 'Confirmed', 'Active', 'Deaths', 'Recovered']].reset_index(drop=True)
from IPython.display import display, HTML

display(HTML(df1.to_html()))
ab=prediction[prediction['ds']>=max(data['ObservationDate'])][['ds','yhat']].reset_index(drop=True)

ab.rename(columns={'ds':'date','yhat':'confirmed_val'},inplace=True)

ab
cd=pd.DataFrame(ab)

cd.to_csv('covid_19_data.csv',index=False)