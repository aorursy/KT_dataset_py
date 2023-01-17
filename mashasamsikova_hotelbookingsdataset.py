from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score

import pandas as pd

import numpy as np

import random as rd

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline
data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

data.head()
months = {'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4, 'May' : 5, 'June' : 6, 'July' : 7, \

          'August' : 8, 'September' : 9, 'October' : 10, 'November' : 11, 'December' : 11}



def change_month(month):

    return months[month]

data['arrival_date_month'] = data['arrival_date_month'].apply(change_month)
data.head()
data.isna().sum()
countries = data['country'].unique()



country_to_num = {}

for i in range(len(countries)):

    if not pd.isnull(countries[i]):

        country_to_num.update({countries[i] : i})   



def change_country(country):

    while pd.isnull(country):

        country = countries[rd.randint(0, 177)]

    return country_to_num[country]    

        
data['country'] = data['country'].apply(change_country)

data.head()
deposits = data['deposit_type'].unique()

deposits_to_ints = {}

for i in range(3):

    deposits_to_ints.update({deposits[i] : i})



data['deposit_type'] = data['deposit_type'].apply(lambda x : deposits_to_ints[x])    
data = data.drop(['company' , 'agent', 'reservation_status_date'], axis=1)

data
data = data.fillna({'children' : data.children.mean()})
data = pd.get_dummies(data)
y = data['is_canceled']

X = data.drop('is_canceled', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
log_reg = LogisticRegression(penalty='l2', C=1.1, max_iter=1000)

log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)