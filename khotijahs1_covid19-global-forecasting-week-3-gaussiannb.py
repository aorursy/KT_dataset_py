import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
train.info()

train[0:10]
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
train = train[['Country_Region','Date','ConfirmedCases','Fatalities']]

train.head()
#Country_Region top 30

train.Country_Region.value_counts()[0:30].plot(kind='bar')

plt.show()
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

train.groupby('Country_Region').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'].plot('bar', color='r',width=0.3,title='Country Region Confirmed Cases', fontsize=10)

plt.xticks(rotation = 90)

plt.ylabel('Confirmed Cases')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)

print(train.groupby('Country_Region').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[1,2]])

print(train.groupby('Country_Region').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[4,5,6]])
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

train.groupby('Country_Region').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'].plot('bar', color='r',width=0.3,title='Country Region Fatalities', fontsize=8)

plt.xticks(rotation = 90)

plt.ylabel('Confirmed Cases')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)

print(train.groupby('Country_Region').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'][[1,2]])

print(train.groupby('Country_Region').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'][[4,5,6]])
print("Any missing sample in training set:",train.isnull().values.any())

print("Any missing sample in test set:",test.isnull().values.any(), "\n")
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



#Singapore

ConfirmedCases_date_Singapore = train[train['Country_Region']=='Singapore'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Singapore = train[train['Country_Region']=='Singapore'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Singapore = ConfirmedCases_date_Singapore.join(fatalities_date_Singapore)







plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_Indonesia.plot(ax=plt.gca(), title='Indonesia')

plt.ylabel("Confirmed  cases", size=13)



plt.subplot(2, 2, 2)

total_date_Malaysia.plot(ax=plt.gca(), title='Malaysia')



plt.subplot(2, 2, 3)

total_date_Thailand.plot(ax=plt.gca(), title='Thailand')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(2, 2, 4)

total_date_Singapore.plot(ax=plt.gca(), title='Singapore')

#Vietnam

ConfirmedCases_date_Vietnam = train[train['Country_Region']=='Vietnam'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Vietnam = train[train['Country_Region']=='Vietnam'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Vietnam = ConfirmedCases_date_Vietnam.join(fatalities_date_Vietnam)





#Philippines

ConfirmedCases_date_Philippines = train[train['Country_Region']=='Philippines'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Philippines = train[train['Country_Region']=='Philippines'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Philippines = ConfirmedCases_date_Philippines.join(fatalities_date_Philippines)



#Cambodia

ConfirmedCases_date_Cambodia = train[train['Country_Region']=='Cambodia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Cambodia = train[train['Country_Region']=='Cambodia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Cambodia = ConfirmedCases_date_Cambodia.join(fatalities_date_Cambodia)



#Laos

ConfirmedCases_date_Laos = train[train['Country_Region']=='Laos'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Laos = train[train['Country_Region']=='Laos'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Laos = ConfirmedCases_date_Laos.join(fatalities_date_Laos)







plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_Vietnam.plot(ax=plt.gca(), title='Vietnam')

plt.ylabel("Confirmed  cases", size=13)



plt.subplot(2, 2, 2)

total_date_Philippines.plot(ax=plt.gca(), title='Philippines')



plt.subplot(2, 2, 3)

total_date_Cambodia.plot(ax=plt.gca(), title='Cambodia')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(2, 2, 4)

total_date_Laos.plot(ax=plt.gca(), title='Laos')



#Brunei

ConfirmedCases_date_Brunei = train[train['Country_Region']=='Brunei'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Brunei = train[train['Country_Region']=='Brunei'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Brunei = ConfirmedCases_date_Brunei.join(fatalities_date_Brunei)





#Myanmar

#ConfirmedCases_date_Myanmar = train[train['Country_Region']=='Myanmar'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

#fatalities_date_Myanmar = train[train['Country_Region']=='Myanmar'].groupby(['Date']).agg({'Fatalities':['sum']})

#total_date_Myanmar = ConfirmedCases_date_Myanmar.join(fatalities_date_Myanmar)









plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_Brunei.plot(ax=plt.gca(), title='Brunei')

plt.ylabel("Confirmed  cases", size=13)



#plt.subplot(2, 2, 2)

#total_date_Myanmar.plot(ax=plt.gca(), title='Myanmar')





train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])

train.info()
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

train.iloc[235:300,:]
test = FunLabelEncoder(test)

test.info()

test.iloc[235:300,:]
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=train["ConfirmedCases"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=train["Fatalities"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
#Select feature column names and target variable we are going to use for training

features= ['Date','Country_Region']

target= 'ConfirmedCases'
#This is input which our classifier will use as an input.

train[features].head(10)
#Display first 10 target variables

train[target].head(100).values
from sklearn.naive_bayes import GaussianNB



# We define the model

nbcla = GaussianNB()



# We train model

nbcla.fit(train[features],train[target])
#Make predictions using the features from the test data set

predictions = nbcla.predict(test[features])



predictions
#Select feature column names and target variable we are going to use for training

features1=['Date','Country_Region']

target1 = 'Fatalities'
#This is input which our classifier will use as an input.

train[features1].head(10)
from sklearn.naive_bayes import GaussianNB



# We define the model

nbcla = GaussianNB()



# We train model

nbcla.fit(train[features1],train[target1])
#Make predictions using the features from the test data set

predictions1 = nbcla.predict(test[features1])



print(predictions1[0:50])
#Create a  DataFrame

submission = pd.DataFrame({'ForecastId':test['ForecastId'],'ConfirmedCases':predictions,'Fatalities':predictions1})

                        



#Visualize the first 10 rows

submission.head(250)
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)