import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

sns.set_context("notebook")



from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot





# Input data files are available in the read-only "../input/" directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_all = pd.read_csv("../input/covid19all/time-series-19-covid-combined_csv.csv")



usa_df = data_all[data_all['Country/Region']=='US']
print(f'There are {usa_df.shape[0]} rows and {usa_df.shape[1]} columns in Daily Covid-19 in USA')

usa_df
print('Daily Covid-19 in USA check\n')

print(usa_df.isna().sum())
# Select Important columns

usa_train= usa_df[['Date', 'Confirmed', 'Recovered', 'Deaths']] 
usa_train.reset_index(drop=True, inplace=True)

usa_train.head()
usa_train.isna().sum()
# Describe the Data

usa_train.describe()
sns.boxplot( x= usa_train.Confirmed)
sns.boxplot( x= usa_train.Recovered)
sns.boxplot( x= usa_train.Deaths)
plt.figure(figsize= (14,8))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Total Confirmed, Recovered, Death in USA" , fontsize = 20)



ax1 = plt.plot_date(data=usa_train,y= 'Confirmed',x= 'Date',label = 'Confirmed',linestyle ='-',color = 'b')

ax2 = plt.plot_date(data=usa_train,y= 'Recovered',x= 'Date',label = 'Recovered',linestyle ='-',color = 'g')

ax3 = plt.plot_date(data=usa_train,y= 'Deaths',x= 'Date',label = 'Death',linestyle ='-',color = 'r')

plt.legend();
df1=usa_train.tail(50)

plt.figure(figsize=(14,8))

sns.barplot(data=df1,x='Date',y='Confirmed',color=sns.color_palette('Set3')[3],label='Confirmed')

sns.barplot(data=df1,x='Date',y='Recovered',color=sns.color_palette('Set3')[4],label='Recovered')

sns.barplot(data=df1,x='Date',y='Deaths',color=sns.color_palette('Set3')[5],label='Deaths')

plt.xlabel('Date')

plt.ylabel('Count')

plt.xticks(rotation = 90)

plt.title("Total Confirmed, Recovered, Death in USA" , fontsize = 20)

plt.legend(frameon=True,fontsize=12);
usa_train.loc[:,["Date","Confirmed"]]
#Model

pred_conf = usa_train.loc[:,["Date","Confirmed"]]

pr_data = pred_conf

pr_data.columns = ['ds','y']

m=Prophet()

m.fit(pr_data)

future=m.make_future_dataframe(periods=22)

forecast=m.predict(future)

import plotly.offline as py



fig = plot_plotly(m, forecast)

py.iplot(fig) 



fig = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')

m.plot_components(forecast)
forecast.to_csv('PredictOutput.csv')
from datetime import datetime, timedelta

fig1 = m.plot(forecast)

datenow = datetime(2020, 7, 27)

dateend = datenow + timedelta(days=21)

datestart = datenow# dateend - timedelta(days=2)

plt.xlim([datestart, dateend])

plt.title("USA COVID-19 forecast", fontsize=20)

plt.xlabel("Date", fontsize=20)

plt.ylabel("Confirmed Count", fontsize=20)

plt.axvline(datenow, color="k", linestyle=":")

plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-20:]