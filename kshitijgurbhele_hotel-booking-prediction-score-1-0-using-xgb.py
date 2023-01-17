# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")

pd.set_option('display.max_columns', None)

df.head()

df.shape
df.isnull().sum()
df.dropna(subset=['country'],inplace=True)

df.dropna(subset=['children'],inplace=True)
df.agent.fillna(0,inplace=True)

df.company.fillna(0,inplace=True)
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.info()
df.describe()
for i in df.columns:

    if len(df[i].unique())<35:

        print('Unique value of ',i,':   ',df[i].unique())
for i in df.columns:

    if len(df[i].unique())<34:

        df.groupby(i)['reservation_status'].count().plot.bar(color='purple')

        plt.show()
df.groupby(['hotel'])['days_in_waiting_list'].count().plot.pie(radius = 1,autopct='%1.1f%%',colors=['yellowgreen', 'lightcoral'])
plt.figure(figsize = (15,7))

sns.set(style="darkgrid")

sns.barplot(x = 'arrival_date_month', y = 'is_canceled',hue='hotel', data = df)
dummy=pd.get_dummies(df[['hotel','meal','country', 'market_segment', 'distribution_channel','reserved_room_type',

       'assigned_room_type','deposit_type','customer_type','reservation_status']],drop_first=True)

pd.set_option('display.max_columns',None)

dummy.head()
from sklearn.preprocessing import LabelEncoder

lc=LabelEncoder()

df['month']=lc.fit_transform(df.arrival_date_month)

df=pd.concat([df,dummy],axis='columns')
df.is_canceled.value_counts()
from sklearn.model_selection import train_test_split

x=df.drop(['is_canceled','arrival_date_month','hotel','meal','country', 'market_segment', 'distribution_channel','reserved_room_type',

       'assigned_room_type','deposit_type','customer_type','reservation_status','reservation_status_date'],axis='columns')

y=df.is_canceled

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
xtrain.shape
xtest.shape
from xgboost import XGBClassifier

xg=XGBClassifier()

xg.fit(xtrain,ytrain)

xg.score(xtest,ytest)
from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()

lg.fit(xtrain,ytrain)

lg.score(xtest,ytest)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(xtrain,ytrain)

rf.score(xtest,ytest)
from sklearn.metrics import classification_report, confusion_matrix 

ytest1=xg.predict(xtest)

print(classification_report(ytest,ytest1))

confusion_matrix(ytest,ytest1)