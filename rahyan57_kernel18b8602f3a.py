# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train= pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
train.head(10)
test.head(10)
print('rows and cols of the train data set',format(train.shape))
print('rows and cols of the test data set',format(test.shape))
train.columns
test.info()
test.describe(include='all').transpose()
train.dtypes[train.dtypes == 'object'] #this describe the which variable is categorical variable
train.info()
train.describe(include='all').transpose()
train['Date'] = train['Date'].astype('datetime64[ns]')
test['Date'] = test['Date'].astype('datetime64[ns]')

print("Train Date type: ", train['Date'].dtype)
print("Test Date type: ",test['Date'].dtype)
print(test['Province/State'].isnull().count())
test1=test.drop("Province/State",axis=1)
print("testing unique :- ",len(test1["Country/Region"].unique()))
test1
print(train['Province/State'].isnull().count())
train1=train.drop("Province/State",axis=1)
print("training unique :- ",len(train1["Country/Region"].unique()))
train1
fig = plt.figure()
ax=fig.add_subplot(111)
ax.plot(train1.groupby('Date')['ConfirmedCases'].sum(),color='blue')
ax.set(title = "rate of coming cov-19 case",ylabel='number of cov-19case',xlabel='Date')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train1.groupby('Date')['Fatalities'].sum(),color='red')
ax.set(title=' graph death rate', ylabel='number of death', xlabel='Date')
plt.show()
countrycase = train.fillna('N/A').groupby(['Country/Region','Province/State'])['ConfirmedCases','Fatalities'].max().sort_values(by='ConfirmedCases',ascending=False)
countrycase.head(10)
countries = countrycase.groupby('Country/Region')['ConfirmedCases','Fatalities'].sum().sort_values(by= 'ConfirmedCases',ascending=False)
countries
countrycase = train.fillna('N/A').groupby(['Country/Region','Province/State'])['ConfirmedCases','Fatalities'].max().sort_values(by='ConfirmedCases',ascending=False)
countrycase.head(10)
countries['country'] = countries.index
countries
trainlong = pd.melt(countries, id_vars=['country'] , value_vars=['ConfirmedCases','Fatalities'])# convert wide to long
trainlong
#Top countries by confirmed cases
topcountries = countries.index[:10]

traintopcountries = trainlong[trainlong['country'].isin(topcountries)]
ax = sns.barplot(x = 'country', hue="variable", y="value", data=traintopcountries)
top10 = train1.groupby('Country/Region')['ConfirmedCases'].sum().sort_values(ascending=False).head(10)

plt.barh(top10.index, top10)
plt.ylabel('Places')
plt.xlabel('Total confirmed cases')
plt.title('Top 10 places with highest confirmed cases')
plt.show()
china_cases = train1[train1['Country/Region'].str.contains('China')][['Date','ConfirmedCases','Fatalities']].reset_index(drop=True)
fig,ax = plt.subplots(2,1, sharex=True)
ax[0].plot(china_cases.groupby('Date')['ConfirmedCases'].sum(), marker='o',color='b', 
            linestyle='--')
ax[1].plot(china_cases.groupby('Date')['Fatalities'].sum(), marker='v',color='r',
            linestyle='--')
ax[0].set_ylabel('Frequency of cases')
ax[1].set_ylabel('Death count')
ax[1].set_xlabel('Date')
plt.xticks(rotation=45)

ax[0].set_title('Total confirmed cases and fatalities in China (Jan 22-Mar 22, 2020)')
plt.show()

restworld_cases = train1[-train1['Country/Region'].str.contains('China')][['Date','ConfirmedCases','Fatalities']].reset_index(drop=True)
fig,ax = plt.subplots(2,1, sharex=True)
ax[0].plot(restworld_cases.groupby('Date')['ConfirmedCases'].sum(), marker='o',color='b', 
            linestyle='--')
ax[1].plot(restworld_cases.groupby('Date')['Fatalities'].sum(), marker='v',color='r',
            linestyle='--')
ax[0].set_ylabel('Frequency of cases')
ax[1].set_ylabel('Death count')
ax[1].set_xlabel('Date')
plt.xticks(rotation=45)

ax[0].set_title('Total confirmed cases and fatalities outside the China (Jan 22-Mar 22, 2020)')
plt.show()
print("date range of train data ",train.Date.min(), train.Date.max())
print("daata range of test data ",test.Date.min(),test.Date.max())
print(trainlong["variable"].value_counts())
statusaffected = trainlong[trainlong['variable']=='ConfirmedCases']
statusaffected.describe().transpose()
statusdead = trainlong[trainlong['variable']=='Fatalities']
statusdead.describe().transpose()
