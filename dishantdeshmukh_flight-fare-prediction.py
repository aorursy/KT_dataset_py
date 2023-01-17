# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')
pd.set_option('display.max_columns',None)
train.head()
train.isnull().sum()
train.dropna(inplace=True)
train.head(1)
train['Journey_data'] = pd.to_datetime(train['Date_of_Journey'], format='%d/%m/%Y').dt.day
train['Journey_month'] = pd.to_datetime(train['Date_of_Journey'],format='%d/%m/%Y').dt.month
train = train.drop('Date_of_Journey',1)
train['hrs'] = pd.to_datetime(train['Dep_Time']).dt.hour
train['min'] = pd.to_datetime(train['Dep_Time']).dt.minute
train = train.drop('Dep_Time',1)
train.head(2)
duration = list(train["Duration"])

for i in range(len(duration)):

    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins

        if "h" in duration[i]:

            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute

        else:

            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []

duration_mins = []

for i in range(len(duration)):

    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration

    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1])) 
train['duration_hrs'] = duration_hours

train['duration_min'] = duration_mins
train['duration_min'].value_counts()
train = train.drop('Duration',1)
train['Airline'].value_counts()
sns.catplot(y='Price',x='Airline',data=train.sort_values('Price',ascending=False),kind='boxen',height=6,aspect=3)
Airline = train['Airline']



Airline = pd.get_dummies(Airline,drop_first=True)
Airline.head()
sns.catplot(y='Price',x='Source',data=train.sort_values('Price',ascending=False),kind='boxen',aspect=3)

plt.show()
Source = train['Source']

Source = pd.get_dummies(Source,drop_first=True)

Source.head()
Destination = train['Destination']

Destination = pd.get_dummies(Destination,drop_first=True)

Destination.head()
train = train.drop(['Route','Additional_Info'],1)
train['Arrival_hours'] = pd.to_datetime(train['Arrival_Time']).dt.hour

train['Arrival_min'] = pd.to_datetime(train['Arrival_Time']).dt.minute



train = train.drop('Arrival_Time',1)
train.head(2)
train['Total_Stops'] = train['Total_Stops'].replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})
train = train.drop(['Airline','Destination','Source'],1)
train_df = pd.concat([train,Airline,Source,Destination],1)
train_df.head()
train_df.shape
X = train.drop('Price',1)

y = train['Price']
from sklearn.ensemble import ExtraTreesRegressor

ect = ExtraTreesRegressor()

ect.fit(X,y)
plt.figure(figsize=(8,8))

feat_imp = pd.Series(ect.feature_importances_,index=X.columns).plot(kind='barh')

plt.show()