import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras 

from keras.layers import Dense

from keras.models import Sequential

from sklearn.model_selection import train_test_split

import time

from datetime import datetime

from sklearn.preprocessing import StandardScaler
train = pd.read_csv('../input/electricity-consumption/train.csv')

test = pd.read_csv('../input/electricity-consumption/test.csv')
print(train.info())

train.head()
test.head(10)
w = train.windspeed

plt.plot(w)

plt.xlabel('samples')

plt.ylabel('windspeed_frequency')

plt.title('Distribution of Windspeed')

plt.show()
print(f'The average wind speed is {round(train.windspeed.mean(),2)}')

print(f'The maximum wind speed is {train.windspeed.max()} and minimum wind speed is {train.windspeed.min()}')
print(f'The average Pressure parameter is {round(train.pressure.mean(),2)}')

print(f'The maximum Pressure value is {train.pressure.max()} and minimum pressure value is {train.pressure.min()}')

plt.scatter(train.pressure,train.electricity_consumption,c='lightblue')

plt.show()
import seaborn as sns

print(train.var2.value_counts())

sns.countplot(x='var2',data=train)
plt.scatter(train.temperature,train.electricity_consumption,c='green')

plt.xlabel('Temperature')

plt.ylabel('Electricity_Consumption')

plt.title('Distribution of Windspeed')

plt.show()
plt.plot(train.electricity_consumption)

plt.xlabel('Samples ---->')

plt.ylabel('Electricity_Consumption')

plt.show()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

train['var2']=LE.fit_transform(train.var2)
print(train.var2.value_counts())

train.head()
test['var2']=LE.fit_transform(test.var2)

test.head()
def datetounix(df):

    # Calling an list of unixtime

    unixtime = []

    

    # Convert Date to seconds

    for date in df['datetime']:

        unixtime.append(time.mktime(date.timetuple()))

    

    # Replacing Date with unixtime list

    df['datetime'] = unixtime

    return(df)
train['datetime'] = pd.to_datetime(train['datetime'])

test['datetime'] = pd.to_datetime(test['datetime'])

test.info()
train.info()
train['Weekday'] = [datetime.weekday(date) for date in train.datetime]

train['Year'] = [date.year for date in train.datetime]

train['Month'] = [date.month for date in train.datetime]

train['Day'] = [date.day for date in train.datetime]

train['Time'] = [((date.hour*60+(date.minute))*60)+date.second for date in train.datetime]

train['Week'] = [date.week for date in train.datetime]

train['Quarter'] = [date.quarter for date in train.datetime]
test['Weekday'] = [datetime.weekday(date) for date in test.datetime]

test['Year'] = [date.year for date in test.datetime]

test['Month'] = [date.month for date in test.datetime]

test['Day'] = [date.day for date in test.datetime]

test['Time'] = [((date.hour*60+(date.minute))*60)+date.second for date in test.datetime]

test['Week'] = [date.week for date in test.datetime]

test['Quarter'] = [date.quarter for date in test.datetime]

test.head()
train.head()
print(train.shape,test.shape)
X_train = train.drop(['ID','electricity_consumption'],axis=1)

X_train.head()
X_train = datetounix(X_train)
X = X_train.values

y = train['electricity_consumption'].values
X
X_test = datetounix(test).drop(['ID'], axis=1).values

X_test
# Standard Scaling

sc = StandardScaler()

X = sc.fit_transform(X)

# Normalizing the target variables

y_train = (y - min(y))/(max(y) - min(y))

y_train
X_test = sc.fit_transform(X_test)
classifier = Sequential()

classifier.add(Dense(units = 10, kernel_initializer = 'uniform',input_dim =13, activation = 'relu'))

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['mae']) # I used both Adam and SGD Optimzer, out of which the loss function was least in Adam.

classifier.fit(X, y_train, batch_size = 16, epochs = 25)
y_pred = classifier.predict(X_test)

y_pred = (y_pred * (max(y) - min(y))) + min(y)



predictions = [int(i) for i in y_pred]



Solution = pd.DataFrame()

Solution['ID'] = test['ID']



# Prepare Solution dataframe

Solution['electricity_consumption'] = predictions

Solution['electricity_consumption'].unique()

Solution
Solution.to_csv('My_submission.csv')