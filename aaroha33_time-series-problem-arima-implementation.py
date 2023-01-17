# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

from datetime import datetime   

from pandas import Series     

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test= pd.read_csv("../input/Test Data Set.csv")

train= pd.read_csv("../input/Train_data set.csv")

sample_submission = pd.read_csv("../input/sample_submission_LSeus50.csv")
train.head()
test.head()
train.info()
test.info()
train.dtypes
train.Datetime = pd.to_datetime(train.Datetime) 

train.dtypes
i = 0

year=[]

day =[]

month=[]

hour=[]

day_of_week=[]



while i < train.shape[0]:

    temp = train.iloc[i,1]

    year.append(temp.year)

    month.append(temp.month)

    day.append(temp.day)

    hour.append(temp.hour)

    day_of_week.append(temp.dayofweek)

    i +=1

#train = data    

train["year"] = year

train["month"]= month

train["day of week"]= day_of_week

train["day"] = day

train["hour"]= hour

train = train.drop("ID",1)
train.head()
graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Year'] = train.year

sns.set_style("whitegrid")

ax = sns.barplot(x="Year", y="Count", data=graph).set_title("Caputuring the Trend")
graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Month'] = train.month



sns.set_style("whitegrid")

ax = sns.barplot(x="Month", y="Count", data=graph).set_title("Trends in Months")

graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Day'] = train.day



sns.set_style("whitegrid")

ax = sns.pointplot(x="Day", y="Count", data=graph).set_title("Trends in Days")
graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Month'] = train.month



sns.set_style("whitegrid")

ax = sns.pointplot(x="Month", y="Count", data=graph).set_title("Trends in Month in another Graph")
sns.set(rc={'figure.figsize':(16.7,6.27)})
graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Day of Week'] = train["day of week"]

fig, axs = plt.subplots(2,1)

sns.pointplot(x="Day of Week", y="Count", data=graph,ax=axs[0]).set_title("Trends in WeekDays")

sns.barplot(x="Day of Week", y="Count", data=graph,ax=axs[1]).set_title("Trends in WeekDays")

plt.show()
graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Hour'] = train.hour



ax = sns.pointplot(x="Hour", y="Count", data=graph).set_title("Trends in Hour")
#train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

train.Timestamp = pd.to_datetime(train.Datetime) 

train.index = train.Timestamp



hourly = train.resample('H').mean()

daily = train.resample('D').mean()

weekly = train.resample('W').mean()

monthly = train.resample('M').mean()

hourly
fig, axs = plt.subplots(3,1)



sns.set(rc={'figure.figsize':(16.7,10.27)})

# We are not going with the hourly for overfitting value

#sns.pointplot(data=hourly,y="Count",x=hourly.index,ax=axs[0]).set_title("Hourly")



sns.pointplot(data=daily,y="Count",x=daily.index,ax=axs[0]).set_title("Daily")

sns.pointplot(data=weekly,y="Count",x=weekly.index,ax=axs[1]).set_title("Weekly")

sns.pointplot(data=monthly,y="Count",x=monthly.index,ax=axs[2]).set_title("Monthly")

plt.show()
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp



# Converting to daily mean

train = train.resample('D').mean()

train.to_csv("train_eda.csv")

train.head()
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
train['Count'].head()
#Plotting data

train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)

#test.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)

plt.show()
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10, 6
train= pd.read_csv("../input/Train_data set.csv")
train.info()
train['Datetime']=pd.to_datetime(train['Datetime'],infer_datetime_format=True)

train = train.drop("ID",1)

train.head()
train.isnull().sum()
#train.Datetime=train.Datetime.apply(lambda x:train.Datetime.strptime(x, '%d.%m.%Y'))
## plot graph

plt.xlabel('Datetime')

plt.ylabel('Count')

#plt.plot(train['Count', 'Datetime'])