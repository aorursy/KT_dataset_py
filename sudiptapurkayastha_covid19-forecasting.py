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

import os

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
train_df
test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
test_df
submission_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
submission_df
print("Total number of countries in this dataset:", train_df.Country_Region.nunique())

print("Date range is from ",train_df.Date.min(), "to" , train_df.Date.max())
cummulative=train_df.groupby(by='Country_Region')[['ConfirmedCases','Fatalities']].max().reset_index()

plt.figure(figsize=(20,10))

sns.barplot(x='ConfirmedCases',y='Country_Region', data=cummulative[cummulative['ConfirmedCases']!=0].sort_values(by='ConfirmedCases',ascending=False).head(50))
plt.figure(figsize=(20,10))

sns.barplot(x='Fatalities',y='Country_Region', data=cummulative[cummulative['Fatalities']!=0].sort_values(by='Fatalities',ascending=False).head(50))
Province_State=train_df['Province_State'].unique()

Province_State=test_df['Province_State'].unique()
dates=train_df['Date'].unique()

dates=test_df['Date'].unique()
Country_Region=train_df['Country_Region'].unique()

Country_Region=test_df['Country_Region'].unique()
train_df['Date']=pd.to_datetime(train_df['Date'])

test_df['Date']=pd.to_datetime(test_df['Date'])
y1_Train = train_df.iloc[:, -2]

y1_Train.head()
y2_Train = train_df.iloc[:, -1]

y2_Train.head()
X_Train = train_df.copy()
X_Train.head()
X_Test = test_df.copy()
X_Test.head()
nonfatalities_train_df=train_df[train_df['Fatalities'] !=0]

nonfatalities_train_df[['Country_Region','Date','ConfirmedCases','Fatalities']]
nonfatalities_train_df=train_df[train_df['Fatalities'] ==0]

nonfatalities_train_df[['Country_Region','Date','ConfirmedCases','Fatalities']]
print(train_df[~train_df['Province_State'].isnull()]['Country_Region'].value_counts())
print(train_df[train_df['Province_State'].isnull()]['Country_Region'].value_counts())

train_df=train_df.fillna('0')
print(test_df[~test_df['Province_State'].isnull()]['Country_Region'].value_counts())
print(test_df[test_df['Province_State'].isnull()]['Country_Region'].value_counts())

test_df=test_df.fillna('0')
from sklearn import preprocessing

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import GaussianNB

le = LabelEncoder()

train_df['Date']=pd.to_datetime(train_df['Date']).dt.strftime("%m%d").astype(int)

train_df['Date']-=122

test_df['Date']=pd.to_datetime(test_df['Date']).dt.strftime("%m%d").astype(int)

test_df['Date']-=122
train_df.Province_State=le.fit_transform(train_df.Province_State)

train_df.Country_Region= le.fit_transform(train_df.Country_Region)
train_df
test_df.Province_State=le.fit_transform(test_df.Province_State)

test_df.Country_Region= le.fit_transform(test_df.Country_Region)
test_df
X=train_df[['Province_State','Country_Region','Date']]

y=train_df[['ConfirmedCases','Fatalities']]

classifier=BinaryRelevance(GaussianNB())

classifier.fit(X,y[['Fatalities']])

pred_fatalities=classifier.predict(test_df[['Province_State','Country_Region','Date']])

classifier.fit(X,y[['ConfirmedCases']])

pred_confirmedcases=classifier.predict(test_df[['Province_State','Country_Region','Date']])
output_confirmedcases= pd.DataFrame(data=pred_confirmedcases.toarray())

output_fatalities= pd.DataFrame(data=pred_fatalities.toarray())



output_confirmedcases=output_confirmedcases.rename(columns={0:"confirmedcases"})

output_fatalities=output_fatalities.rename(columns={0:"fatalities"})
test_df.ForecastId
result=pd.concat([test_df.ForecastId,output_confirmedcases,output_fatalities],axis=1)

result
result.to_csv('submission.csv',index=False)