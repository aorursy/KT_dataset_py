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
import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

df.set_index('hotel',inplace = True)

df.head()
df.columns
df.isna().sum()
df.drop(['company', 'agent', 'country'],inplace=True,axis = 1)

df.head()
df['children'].mode()
df['children'].fillna(0,inplace =True)
encode = LabelEncoder()

df['arrival_date_month'] = encode.fit_transform(df['arrival_date_month'])

df['meal'] = encode.fit_transform(df['meal'])

df['market_segment'] = encode.fit_transform(df['market_segment'])

df['distribution_channel'] = encode.fit_transform(df['distribution_channel'])

df['reserved_room_type'] = encode.fit_transform(df['reserved_room_type'])

df['assigned_room_type'] = encode.fit_transform(df['assigned_room_type'])

df['deposit_type'] = encode.fit_transform(df['deposit_type'])

df['customer_type'] = encode.fit_transform(df['customer_type'])

df['reservation_status'] = encode.fit_transform(df['reservation_status'])
df['arrival_date_year'] = df['arrival_date_year'].map({2015:1, 2016:2, 2017:3})
scaler = MinMaxScaler()

df['lead_time'] = scaler.fit_transform(df['lead_time'].values.reshape(-1,1))

df['adr'] = scaler.fit_transform(df['adr'].values.reshape(-1,1))
df.corr()
plt.figure(figsize = (10,10))

sns.heatmap(df.corr())
data = df[['reservation_status','total_of_special_requests','required_car_parking_spaces','deposit_type','booking_changes','assigned_room_type','previous_cancellations','distribution_channel','lead_time','is_canceled']]
X = data.drop(['is_canceled'],axis= 1)

y = data['is_canceled']
linreg = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)

linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
accuracy = linreg.score(X_test,y_test)

print(accuracy)
matrix = confusion_matrix(y_test,y_pred.round())

matrix
logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = logreg.score(X_test,y_test)

print(accuracy)
matrix = confusion_matrix(y_test, y_pred.round())

matrix
df1 = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
plt.figure(figsize = (10,5))

df1.groupby(['arrival_date_month'])['arrival_date_month'].count().plot.bar()
plt.figure(figsize = (10,5))

sns.countplot(x='arrival_date_year',hue = 'hotel',data=df1)
plt.figure(figsize=(30,5))

df1.groupby(['country'])['country'].size().sort_values(ascending=False).plot.bar()
maxpop = df1[df1['country'] == 'PRT']

plt.figure(figsize = (10,5))

maxpop.groupby(['arrival_date_month'])['arrival_date_month'].count().plot.bar()
plt.figure(figsize = (10,5))

sns.countplot(x='arrival_date_year',hue='hotel',data=maxpop)
plt.figure(figsize = (10,5))

sns.countplot(x='market_segment',hue='hotel',data=df1)
plt.figure(figsize = (10,5))

sns.countplot(x='is_canceled',hue='hotel',data=df1)
change_room = df1[df1['reserved_room_type'] != df1['assigned_room_type']]
plt.figure(figsize=(10,5))

sns.countplot(x='is_canceled',hue='hotel',data=change_room)
plt.figure(figsize=(10,5))

deposit =df1.groupby(['deposit_type','is_canceled'])['deposit_type'].count()

print(deposit)

sns.countplot(x=df1['deposit_type'],data=df1,hue='is_canceled')
plt.figure(figsize=(10,5))

customer = df1.groupby(['customer_type','is_canceled'])['customer_type'].count()

print(customer)

sns.countplot(x='customer_type',hue='is_canceled',data=df1)