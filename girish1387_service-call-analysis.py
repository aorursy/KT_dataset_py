# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
serviceCall = pd.read_csv(r"../input/police-department-calls-for-service.csv")
serviceCall.head()
print("Missing values: " , serviceCall.isnull().sum())
print("Unique values: " , serviceCall.nunique())

serviceCall['Year'] = pd.DatetimeIndex(serviceCall['Call Date']).year
serviceCall['MonthYear'] = pd.DatetimeIndex(serviceCall['Call Date']).strftime("%Y-%m")
serviceCall.groupby(serviceCall['Year'])['Crime Id'].count().plot.bar()
from pandas.api.types import CategoricalDtype
cats = [ 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
        'October', 'November', 'December']
cat_type = CategoricalDtype(categories=cats, ordered=True)
serviceCall['Month'] = pd.DatetimeIndex(serviceCall['Call Date']).strftime("%B")
serviceCall.groupby(serviceCall['Month'])['Crime Id'].count().reindex(cats).plot.bar()
serviceCall.groupby(serviceCall['MonthYear'])['Crime Id'].count().plot.box()
serviceCall.groupby(serviceCall['MonthYear'])['Crime Id'].count().describe()
serviceCall.groupby(serviceCall['MonthYear'])['Crime Id'].count().plot.line()
serviceCall.groupby(serviceCall['Call Date'])['Crime Id'].count().plot.box()
serviceCall.groupby(serviceCall['Call Date'])['Crime Id'].count().plot.line()
#There are unusally few calls on few days
serviceCallDesc = serviceCall.groupby(serviceCall['Call Date'])['Crime Id'].count().describe()
serviceCallDesc
#Lets find the day where calls are very low.
#Aussuming normal distribution, lets see the days where call are too low. ie < 2.5%
lowCalls = serviceCall.groupby(serviceCall['Call Date'])['Crime Id'].count()
lowCalls[lowCalls < (serviceCallDesc['mean'] - (3 * serviceCallDesc['std']))]
#Thanks Giving and Christmas have very low service calls
serviceCall['Weekday'] = pd.DatetimeIndex(serviceCall['Call Date']).weekday_name
cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
cat_type = CategoricalDtype(categories=cats, ordered=True)
serviceCall.groupby(serviceCall['Weekday'])['Crime Id'].count().reindex(cats).plot.bar()
#Marginal dip in service call on Sundays
serviceCall['Hour'] = pd.DatetimeIndex(serviceCall['Call Time']).hour
serviceCall.groupby(serviceCall['Hour'])['Crime Id'].count().plot.bar()
#High Traffic during 4 PM to 7 PM
serviceCall.groupby(['Address']).size().sort_values(ascending=False).head(10)
serviceCall[serviceCall['Hour'].isin([17,18,19])].groupby(['Address']).size().sort_values(ascending=False).head(10)
#300 Block Of Market St recives high no of calls during peak hour
serviceCall.groupby(['Common Location']).size().sort_values(ascending=False).head(10)
serviceCall[serviceCall['Hour'].isin([17,18,19])].groupby(['Common Location']).size().sort_values(ascending=False).head(10)
#Call traffic from Embarcadero Muni Station, Sf increases during peak hour
serviceCall.groupby(['Original Crime Type Name']).size().sort_values(ascending=False).head(10)
percent = serviceCall.groupby(['Original Crime Type Name']).size().sort_values(ascending=False).head(10) / serviceCall['Crime Id'].count() 
percent.sum() * 100
#Top 10 crime types constitue upto 50% of the service calls. There are 18521 Crime Types reported so far
serviceCall[serviceCall['Hour'].isin([17,18,19])].groupby(['Original Crime Type Name']).size().sort_values(ascending=False).head(10)
#Muni Inspections rise during peak hour
serviceCall.groupby(['Disposition']).size().sort_values(ascending=False)
serviceCall[serviceCall['Hour'].isin([17,18,19])].groupby(['Disposition']).size().sort_values(ascending=False)
# Summary.
# The daily average call is 2264 with Standard deviation of 215. 
# The peak is during 4 to 7 pm
# A very slight dip in the no of calls on Sundays
# Signfiacant increase in calls from 300 Block Of Market St during peak hours
# Signfiacant increase in calls from Embarcadero Muni Station, Sf during peak hours
# Top crime types are Passing Call, Traffic Stop, Suspicious Person, Homeless Complaint.
# Top 10 crime types constitue upto 50 % of the reported crimes