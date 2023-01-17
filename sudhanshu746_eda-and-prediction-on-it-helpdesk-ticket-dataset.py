# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Complete in 45 min start-10:49
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.ticker as mtic
import matplotlib.pyplot as plot
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
ticketDf=pd.read_csv('../input/WA_Fn-UseC_-IT-Help-Desk.csv')
# Data screening 
ticketDf.head(20)
#ticketDf.isna().any()
#ticketDf.dtypes

'''
# Exploration Data Analysis EDA
# Now Lets check correlation of Churn with other variables Correlation
'''
#print(ticketDf.isnull().sum())
'''
ticketDf['Satisfaction'].replace(to_replace='0 - Unknown',value=0,inplace=True)
ticketDf['Satisfaction'].replace(to_replace='1 - Unsatisfied',value=0,inplace=True)
ticketDf['Satisfaction'].replace(to_replace='2 - Satisfied',value=1,inplace=True)
ticketDf['Satisfaction'].replace(to_replace='3 - Highly satisfied',value=1,inplace=True)
'''
ticketDf['Satisfaction'].replace(to_replace='0 - Unknown',value=0,inplace=True)
ticketDf['Satisfaction'].replace(to_replace='1 - Unsatisfied',value=1,inplace=True)
ticketDf['Satisfaction'].replace(to_replace='2 - Satisfied',value=2,inplace=True)
ticketDf['Satisfaction'].replace(to_replace='3 - Highly satisfied',value=3,inplace=True)

ticketDf['Severity'].replace(to_replace='0 - Unclassified',value=0,inplace=True)
ticketDf['Severity'].replace(to_replace='1 - Minor',value=1,inplace=True)
ticketDf['Severity'].replace(to_replace='2 - Normal',value=2,inplace=True)
ticketDf['Severity'].replace(to_replace='3 - Major',value=3,inplace=True)
ticketDf['Severity'].replace(to_replace='4 - Critical',value=4,inplace=True)


ticketDf["daysOpen"] = pd.cut(ticketDf["daysOpen"],bins=5)

dummiesDf = pd.get_dummies(ticketDf)
dummiesDf.head(30)
plot.figure(figsize=(15,10))
dummiesDf.corr()['Satisfaction'].sort_values(ascending=False).plot(kind='bar')

# Conclusion: as per correlation, Severty_3 major, ITowner, Severty_4 critical  positively correlated with Satisfaction
# Whereas, DaysOpen and Severty_2 Minor negetively correlated with Satisfaction

# Lets check relation between Days of ticket Open and Satisfaction
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8']

ds = ticketDf.groupby(['Satisfaction','daysOpen']).size().unstack()
ax = (ds.T*100.0 / ds.T.sum()).T.plot(kind='bar',width = 0.2,stacked = True,rot = 0, figsize = (8,6),color = colors)

# Conclusion : Number of ticket opened days can fluctuate Satisfaction level
# Lets check relation between Severity  and Satisfaction
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8']

ds = ticketDf.groupby(['Satisfaction','Severity']).size().unstack()
ax = (ds.T*100.0 / ds.T.sum()).T.plot(kind='bar',width = 0.2,stacked = True,rot = 0, figsize = (8,6),color = colors)

# Conclusion : Where most of tickets having Severity-2 common for all Satisfaction levels 
# Features selection


dummiesDf = pd.get_dummies(ticketDf)
'''
selected_features =['Severity',
       'RequestorSeniority_1 - Junior', 'RequestorSeniority_2 - Regular',
       'RequestorSeniority_3 - Senior', 'RequestorSeniority_4 - Management',
       'FiledAgainst_Access/Login', 'FiledAgainst_Hardware',
       'FiledAgainst_Software', 'FiledAgainst_Systems', 'TicketType_Issue',
       'TicketType_Request', 'Priority_0 - Unassigned', 'Priority_1 - Low',
       'Priority_2 - Medium', 'Priority_3 - High', 'daysOpen_(-0.054, 10.8]',
       'daysOpen_(10.8, 21.6]', 'daysOpen_(21.6, 32.4]',
       'daysOpen_(32.4, 43.2]', 'daysOpen_(43.2, 54.0]']

selected_features =['Severity','ITOwner','RequestorSeniority_4 - Management','RequestorSeniority_3 - Senior','Priority_3 - High', 'daysOpen_(-0.054, 10.8]',
       'daysOpen_(10.8, 21.6]', 'daysOpen_(21.6, 32.4]',
       'daysOpen_(32.4, 43.2]', 'daysOpen_(43.2, 54.0]']
'''
selected_features =['Severity','ITOwner','daysOpen_(43.2, 54.0]']
#print(dummiesDf.columns)
X = dummiesDf[selected_features]
#X = dummiesDf.drop(columns = ['Satisfaction','ticket','requestor'])
Y = dummiesDf['Satisfaction'].values
X.head(20)
# Lets scale all the variables from a range of 0 to 1
# Transforms features by scaling each feature to a given range.
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

X.head(20)
'''
1. Let's use KNN classifier to approach Telecom churn data
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=102)
knModel = KNeighborsClassifier(n_neighbors=11)
knModel.fit(x_train,y_train)
testPrediction =  knModel.predict(x_test)
print(metrics.accuracy_score(y_test,testPrediction))
from sklearn.metrics import classification_report
print(classification_report(y_test, testPrediction))