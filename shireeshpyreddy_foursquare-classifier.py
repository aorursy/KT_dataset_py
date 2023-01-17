# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ny_data = pd.read_csv("../input/dataset_TSMC2014_NYC.csv")

ny_data.head()
ny_data.describe()
import datetime

data = []

for utcoffset_index, utcoffset in enumerate(ny_data['utcTimestamp']):    

    year = datetime.datetime.strptime(utcoffset, '%a %b %d %X %z %Y').strftime('%Y')

    month = datetime.datetime.strptime(utcoffset, '%a %b %d %X %z %Y').strftime('%m')

    day = datetime.datetime.strptime(utcoffset, '%a %b %d %X %z %Y').strftime('%d')

    weekday = datetime.datetime.strptime(utcoffset, '%a %b %d %X %z %Y').strftime('%a')

    time = datetime.datetime.strptime(utcoffset, '%a %b %d %X %z %Y').strftime('%X')

    data.append([year, month, day, weekday, time])

ny_data_fe = pd.DataFrame(data, columns = ['year', 'month', 'day', 'weekday', 'time'])



for col in ny_data_fe.columns:

    if col not in ['weekday', 'time']:

        ny_data[col] = pd.to_numeric(ny_data_fe[col])

    else:

        ny_data[col] = ny_data_fe[col]
ny_data.describe()
# ny_data.drop('utcTimestamp', 1, inplace=True)

ny_data['year'] = np.where(ny_data['year'] == 2012, 1, 0)

ny_data.head()
grouped = ny_data.groupby(by='venueCategoryId')



for _, group in grouped:

    print(group.head())
selected_features = ['latitude', 'longitude', 'timezoneOffset']
data, labels = ny_data[selected_features], ny_data['venueCategory']

data.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, random_state=42)



X_train.shape, X_test.shape, y_test.shape, y_train.shape
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()



clf.fit(X_train, y_train)

clf.score(X_train, y_train)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



predictions = clf.predict(X_test)



print(classification_report(y_test, predictions))
clf.score(X_test, y_test)