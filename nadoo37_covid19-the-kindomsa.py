# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.colors as mcolors

import datetime 
import seaborn as sns
import matplotlib.pyplot as plt
#plt.style.use('seaborn')
%matplotlib inline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

covid_data=pd.read_csv("../input/covid19-saudiarabia/saudi_covid19_places.csv")
covid_data.tail()
covid_data.head()
covid_data.shape
covid_data.columns
covid_data['DateTime']
covid_data['DateTime'].dtype
covid_data['Date']=pd.to_datetime(covid_data['DateTime'], format = "%Y-%m-%d ")
covid_data['Date'].head()
# missing value check
missing_values_count = covid_data.isnull().sum()
print(missing_values_count)
covid_data.describe()
import plotly.express as px
# Top 50 cities with highest confirmed cases
covid_city_top50=covid_data.sort_values("Confirmed",ascending=False).head(50)

fig = px.bar(covid_city_top50, 
             x="Place_EN",
             y="Confirmed",
             orientation='v',
             height=800,
             title='Top 50 cities with COVID19 Confirmed Cases',
            color='Place_EN')
fig.show()
import plotly.express as px
# Top 50 cities with highest Recovered 
covid_city_top50=covid_data.sort_values("Recovered",ascending=False).head(50)

fig = px.bar(covid_city_top50, 
             x="Place_EN",
             y="Recovered",
             orientation='v',
             height=800,
             title='Top 50 cities with COVID19 Recovered Cases',
            color='Place_EN')
fig.show()
import plotly.express as px
# Top 50 cities with highest Deaths 
covid_city_top50=covid_data.sort_values("Deaths",ascending=False).head(50)

fig = px.bar(covid_city_top50, 
             x="Place_EN",
             y="Deaths",
             orientation='v',
             height=800,
             title='Top 50 cities with COVID19 Deaths Cases',
            color='Place_EN')
fig.show()
# Set the width and height of the figure
plt.figure(figsize=(20,15))

# Add title
plt.title("Cumulative cases by date")

# Line chart showing daily streams of 'Cases'
sns.lineplot(data=covid_data['Confirmed'],color='purple', label="Confirmed")
sns.lineplot(data=covid_data['Deaths'],color='red', label="Deaths", linestyle='--')
sns.lineplot(data=covid_data['Recovered'],color='green', label="Recovered")

# Add label for horizontal axis
plt.xlabel("DateTime")
plt.show()

boxplot=covid_data.boxplot(grid=False, rot=45, figsize=(10,8))
boxplot
boxplot=covid_data.boxplot(grid=False, column=['Recovered'],figsize=(10,8))
boxplot

Q1=covid_data['Recovered'].quantile(0.25)
Q3=covid_data['Recovered'].quantile(0.75)
print(Q1, '   ', Q3)
#THE Result
# Q1 is 25% 
# Q3 is 75% 
#interquartile range (IQR)
IQR= Q3-Q1
print("interquartile range is ", IQR)
minimum=Q1 - 1.5*IQR
maximum=Q3 + 1.5*IQR
print ( "minimum is ", minimum,'     ', "and ","maximum is ",maximum )
#covid_data['Confirmed_outlier', 'Deaths_outlier','Active_outlier']=False
covid_data['Recovered_outlier']=False

for index, row in covid_data.iterrows():
   if row ['Recovered']> maximum:
     covid_data.at[index,'Recovered_outlier']=True
covid_data['Recovered_outlier'].sum()
mean=covid_data['Recovered_outlier'].mean()
mean
for index, row in covid_data.iterrows():
    if row ['Recovered_outlier']==True:
        covid_data.at[index,'Recovered']=mean
boxplot2=covid_data.boxplot(grid=False, fontsize=15, column=['Recovered'], figsize=(10,8))
boxplot2
Q1_confirmed=covid_data['Confirmed'].quantile(0.25)
Q3_confirmed=covid_data['Confirmed'].quantile(0.75)
print(Q1_confirmed, '   ', Q3_confirmed)
#THE Result
# Q1 is 25% 
# Q3 is 75% 
#interquartile range (IQR)
IQR_confirmed= Q3_confirmed-Q1_confirmed
print("interquartile range is ", IQR_confirmed)
minimum_confirmed=Q1_confirmed - 1.5*IQR_confirmed
maximum_confirmed=Q3_confirmed + 1.5*IQR_confirmed
print ( "minimum is ", minimum_confirmed,'     ', "and ","maximum is ",maximum_confirmed )
#_confirmed
#covid_data['Confirmed_outlier', 'Deaths_outlier','Active_outlier']=False
covid_data['Confirmed_outlier']=False

for index, row in covid_data.iterrows():
   if row ['Confirmed']> maximum:
     covid_data.at[index,'Confirmed_outlier']=True
sum__confirmed=covid_data['Confirmed_outlier'].sum()
sum__confirmed
mean_confirmed=covid_data['Confirmed_outlier'].mean()
mean_confirmed
for index, row in covid_data.iterrows():
    if row ['Confirmed_outlier']==True:
        covid_data.at[index,'Confirmed']=mean_confirmed
boxplot_confirmed=covid_data.boxplot(grid=False, fontsize=15, column=['Confirmed'], figsize=(10,8))
boxplot_confirmed
#outliers = covid_data['Confirmed','Deaths', 'Recovered','Active'] 

###################################################
#Grouping different cases as per the date
covid_sa = covid_data.groupby(['Date'])[['Confirmed','Deaths','Recovered', 'Active']].sum()
covid_sa
#Grouping different cases as per the date
covid_sa.describe()                                     
covid_sa.plot(kind='line', figsize=(20,8))
total_=covid_data['Confirmed'].sum()
total_
total_=covid_data['Recovered'].sum()
total_
total_=covid_data['Deaths'].sum()
total_
###&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
from fbprophet import Prophet
model2 = Prophet()
cases=covid_sa.reset_index()
cases.head()
#[['Date''Confirmed','Deaths','Recovered', 'Active'
comfired_cases=cases[['Date','Confirmed']]
recovered_cases=cases[['Date','Recovered']]
death_cases=cases[['Date','Deaths']]
Active_cases=cases[['Date','Active']]

#Recovered
recovered_cases
death_cases

recovered_cases.rename(columns={'Date':'ds','Recovered':'y'}, inplace=True)


#2

train2=recovered_cases[:40]
test2=recovered_cases[40:]
train2.head()
#Confirmed
recovered_cases.tail()
#fit fp model
model2.fit(train2)
dates_in_future2= model2.make_future_dataframe(periods=190)
#dates_in_future2= model2.make_future_dataframe(periods=195)
dates_in_future2.tail()
predicted2=model2.predict(dates_in_future2)
model2.plot(predicted2)
plt.title("Number of death cases over time")
plt.xlabel('death cases for coming days')
plt.ylabel('Number of death cases')
#observe covid 19 separately: Daily yearly and weekly seasonality.
model2.plot_components(predicted2)
#predicted
predicted2.tail()
predicted2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#compares to  actual values using a few different metrics - R-Squared and Mean Squared Error (MSE).
metric_covid2 = predicted2.set_index('ds')[['yhat']].join(recovered_cases.set_index('ds').y).reset_index()
metric_covid2.tail()

#11
metric_covid2.dropna(inplace=True)
metric_covid2.tail()

print ('R-Squared value: ',r2_score(metric_covid2.y, metric_covid2.yhat))
print ('Mean Square Error value: ',mean_squared_error(metric_covid2.y, metric_covid2.yhat))
print ('Mean Absolute Error value: ',mean_absolute_error(metric_covid2.y, metric_covid2.yhat))
y = metric_covid2.y.values
y_pred = metric_covid2.yhat.values
# plot expected vs actual
plt.plot(metric_covid2.y, label='Actual')
plt.plot(metric_covid2.yhat, label='Predicted')
plt.title('Comparison between Actual and predicted value')
plt.legend()
plt.show()