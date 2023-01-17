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

import pandas as pd

import matplotlib.pyplot as plt
train=pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
test1=test
train.head(5)
test1.head(5)
test.head(5)
train.info()
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'].plot('bar', color='r',width=0.3,title='Date Confirmed Cases', fontsize=10)

plt.xticks(rotation = 90)

plt.ylabel('Date')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)

print(train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[1,2]])

print(train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[4,5,6]])
#Country_Region top 30

train.Country_Region.value_counts()[0:30].plot(kind='bar')

plt.show()
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

train.groupby('Date').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'].plot('bar', color='r',width=0.3,title='Date Fatalities Cases', fontsize=10)

plt.xticks(rotation = 90)

plt.ylabel('Date')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)

print(train.groupby('Date').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'][[1,2]])

print(train.groupby('Date').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'][[4,5,6]])
#visualization of main places US,ITALY,CHINA,UK

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







plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_US.plot(ax=plt.gca(), title='US')

plt.ylabel("Confirmed  cases", size=13)



plt.subplot(2, 2, 2)

total_date_China.plot(ax=plt.gca(), title='China')



plt.subplot(2, 2, 3)

total_date_Italy.plot(ax=plt.gca(), title='Italy')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(2, 2, 4)

total_date_Australia.plot(ax=plt.gca(), title='Australia')
train['Date']= pd.to_datetime(train['Date']) 

test['Date']= pd.to_datetime(test['Date'])
train = train.set_index(['Date'])

test = test.set_index(['Date'])
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
create_time_features(train).head()

create_time_features(test).head()
train.head(5)
train.drop("date", axis=1, inplace=True)

test.drop("date", axis=1, inplace=True)
confirmed_corr=train.corr()['ConfirmedCases']
confirmed_corr.sort_values(ascending=False)
fatalities_corr=train.corr()['Fatalities']
fatalities_corr.sort_values(ascending=False)
train.drop("dayofweek", axis=1, inplace=True)

test.drop("dayofweek", axis=1, inplace=True)
train.drop("hour", axis=1, inplace=True)

test.drop("hour", axis=1, inplace=True)
train.drop("quarter", axis=1, inplace=True)

test.drop("quarter", axis=1, inplace=True)
train.drop("year", axis=1, inplace=True)

test.drop("year", axis=1, inplace=True)
train.drop("Province_State", axis=1, inplace=True)

test.drop("Province_State", axis=1, inplace=True)
train.drop("Id", axis=1, inplace=True)

test.drop("ForecastId", axis=1, inplace=True)
test.info()
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
test = FunLabelEncoder(test)

test.info()
test.head(5)
x_train= train[['Country_Region', 'month', 'dayofyear', 'dayofmonth' , 'weekofyear']]

y1 = train[['ConfirmedCases']]

y2 = train[['Fatalities']]

x_test = test[['Country_Region', 'month', 'dayofyear', 'dayofmonth' , 'weekofyear']]
from sklearn.ensemble import RandomForestClassifier



# We define the model

tree_model= RandomForestClassifier(n_estimators=100, max_depth=200,

                        random_state=1)
#for confirmed cases

tree_model.fit(x_train,y1)

prediction1 = tree_model.predict(x_test)

prediction1 = pd.DataFrame(prediction1)

prediction1.columns = ["ConfirmedCases_prediction"]
prediction1.head()
#for fatalities

tree_model.fit(x_train,y2)

prediction2 = tree_model.predict(x_test)

prediction2 = pd.DataFrame(prediction2)

prediction2.columns = ["Death_prediction"]
sub_new=test1[["ForecastId"]]

sub_new


submit = pd.concat([prediction1,prediction2,sub_new],axis=1)

submit.head()
# Clean

submit.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

submit = submit[['ForecastId','ConfirmedCases', 'Fatalities']]



submit["ConfirmedCases"] = submit["ConfirmedCases"].astype(int)

submit["Fatalities"] = submit["Fatalities"].astype(int)
submit.info()
submit.head(5)# Final prediction
submit.shape
submit.to_csv("submission.csv",index=False)