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
import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
train.head()
fig = plt.figure(figsize=(16,8))

axs = fig.add_subplot(111)

plt.scatter(train['Date'], train['ConfirmedCases'])

plt.xticks(rotation = 90)

plt.ylabel('No of COVID19 Cases')

print(train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[1,2]])

print(train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[4,5,6]])
d_train_2 = train.loc[train.groupby('Country_Region')['ConfirmedCases'].idxmax(), :].reset_index()

d_train_2

sorted_train = d_train_2.sort_values('ConfirmedCases', ascending=False)

top10_C = sorted_train.head(10)

top10_Countries = top10_C[['Country_Region','Date', 'ConfirmedCases','Fatalities']]

top10_Countries
# top10_Countries.plot(kind = 'bar')

top10 = pd.DataFrame(top10_Countries)

top10
top10_Countries.plot(kind = 'bar')
# Percentage of Fatalities to Confirmed Cases by Country

top10_Countries['percent'] = top10_Countries['Fatalities'] /top10_Countries['ConfirmedCases'] 

# top10_Countries2 = top10_Countries.drop('Fatalities')

top10_Countries.percent.plot.bar('Country_Regions')
#Count by country

train.Country_Region.value_counts()[0:30].plot(kind='bar')

plt.show()
#US

ConfirmedCases_date_US = train[train['Country_Region']=='US'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_US = train[train['Country_Region']=='US'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_US = ConfirmedCases_date_US.join(fatalities_date_US)





#China

ConfirmedCases_date_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_China = ConfirmedCases_date_China.join(fatalities_date_China)



#Italy

ConfirmedCases_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Italy = ConfirmedCases_date_Italy.join(fatalities_date_Italy)



#Australia

ConfirmedCases_date_Australia = train[train['Country_Region']=='Australia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Australia = train[train['Country_Region']=='Australia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Australia = ConfirmedCases_date_Australia.join(fatalities_date_Australia)



#Indonesia

ConfirmedCases_date_Indonesia = train[train['Country_Region']=='Indonesia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Indonesia = train[train['Country_Region']=='Indonesia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Indonesia = ConfirmedCases_date_Indonesia.join(fatalities_date_Indonesia)





#Malaysia

ConfirmedCases_date_Malaysia = train[train['Country_Region']=='Malaysia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Malaysia = train[train['Country_Region']=='Malaysia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Malaysia = ConfirmedCases_date_Malaysia.join(fatalities_date_Malaysia)



#Thailand

ConfirmedCases_date_Thailand = train[train['Country_Region']=='Thailand'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Thailand = train[train['Country_Region']=='Thailand'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Thailand = ConfirmedCases_date_Thailand.join(fatalities_date_Thailand)



#India

ConfirmedCases_date_India = train[train['Country_Region']=='India'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_India = train[train['Country_Region']=='India'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_India = ConfirmedCases_date_India.join(fatalities_date_India)



plt.figure(figsize = (15,10))

plt.subplot(4, 4, 1)

total_date_US.plot(ax=plt.gca(), title='US')

plt.ylabel("Confirmed  cases", size=13)



plt.subplot(4, 4, 2)

total_date_China.plot(ax=plt.gca(), title='China')



plt.subplot(4, 4, 3)

total_date_Italy.plot(ax=plt.gca(), title='Italy')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(4, 4, 4)

total_date_Australia.plot(ax=plt.gca(), title='Australia')



plt.figure(figsize = (15,10))

plt.subplot(4, 4, 5)

total_date_Indonesia.plot(ax=plt.gca(), title='Indonesia')

plt.ylabel("Confirmed  cases", size=13)



plt.subplot(4, 4, 6)

total_date_Malaysia.plot(ax=plt.gca(), title='Malaysia')



plt.subplot(4, 4, 7)

total_date_Thailand.plot(ax=plt.gca(), title='Thailand')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(4, 4, 8)

total_date_India.plot(ax=plt.gca(), title='India')
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])

train['Date'] = train['Date'].astype('int64')

test['Date'] = test['Date'].astype('int64')

train.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def FunLabelEncoder(df):

    for c in df.columns:

        if df.dtypes[c] == object:

            le.fit(df[c].astype(str))

            df[c] = le.transform(df[c].astype(str))

    return df
train = FunLabelEncoder(train)

train.info()

train.head()
test = FunLabelEncoder(test)

test.info()

test.head()
train_outcome = pd.crosstab(index=train["ConfirmedCases"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
train_outcome = pd.crosstab(index=train["Fatalities"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
features=['Date','Country_Region']

target = 'ConfirmedCases'
train[features].head(10)
from sklearn.ensemble import RandomForestClassifier



# We define the model

rfcla = RandomForestClassifier(n_estimators=50, random_state=42,n_jobs=-1)

# We train model

rfcla.fit(train[features],train[target])
predictions = rfcla.predict(test[features])



predictions
features1=['Date','Country_Region']

target1 = 'Fatalities'
from sklearn.ensemble import RandomForestClassifier



# We define the model

rfcla = RandomForestClassifier(n_estimators=10, random_state=42,n_jobs=-1)

                             

                            

# We train model

rfcla.fit(train[features1],train[target1])
#Make predictions using the features from the test data set

predictions1 = rfcla.predict(test[features1])



print(predictions1[0:50])
#Create a  DataFrame

my_submission = pd.DataFrame({'Id':test.ForecastId,'ConfirmedCases':predictions,'Fatalities':predictions1})

# my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

                             

my_submission.to_csv('submission.csv', index=False)