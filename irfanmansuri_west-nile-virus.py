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
from sklearn import ensemble, preprocessing

train = pd.read_csv("/kaggle/input/predict-west-nile-virus/train.csv.zip")
test = pd.read_csv("/kaggle/input/predict-west-nile-virus/test.csv.zip")
sample = pd.read_csv("/kaggle/input/predict-west-nile-virus/sampleSubmission.csv.zip")
weather = pd.read_csv("/kaggle/input/predict-west-nile-virus/weather.csv.zip")
train.head()
# Get labels
labels = train.WnvPresent.values
labels
weather.head()
train.shape
train.isnull().sum()
test.head()
test.shape
# Ignoring the use of codesum for this benchmark
#weather = weather.drop('CodeSum', axis = 1)
train.info()
X_train = train.loc[:, ['Latitude','Longitude']]
Y_train = train.WnvPresent
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
train_x, val_x, train_y, val_y = train_test_split(X_train, Y_train, test_size = .2, random_state = 9)

model.fit(train_x, train_y)
pred = model.predict(val_x)
#model.score(val_y, pred)
pred.shape
val_y.shape
score = np.mean((pred-val_y)**2)
score
# Not using codesum for this benchmark
weather = weather.drop('CodeSum', axis = 1)
# Split station 1 and 2 and join horizontally
#weather_stn1 = weather[weather['Station'] == 1]
#weather_stn2 = weather[weather['Station'] == 2]
#weather_stn1 = weather_stn1.drop('Station', axis = 1)
#weather_stn2 = weather_stn2.drop('Station', axis = 1)
#weather = weather_stn1.merge(weather_stn2, on = 'Date')
weather
weather_stn1 = weather[weather['Station'] == 1]
weather_stn1
weather_stn2 = weather[weather['Station'] == 2]
weather_stn2
weather_stn1 = weather_stn1.drop('Station', axis = 1)
weather_stn1
weather_stn2 = weather_stn2.drop('Station', axis = 1)
weather_stn2
weather = weather_stn1.merge(weather_stn2, on = 'Date')
weather
weather.info()
# Replace some missing values and T with -1

weather = weather.replace('M', -1)
weather
weather.isnull().sum()
weather = weather.replace('-', -1)
weather
weather = weather.replace('T', -1)
weather
weather = weather.replace(' T', -1)
weather
weather = weather.replace('  T', -1)
weather
# Function to extract month and day from dataset
# You can also use parse_dates of Pandas

def create_month(x):
    return x.split('-')[1]
def create_day(x):
    return x.split('-')[2]
train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)
test['month'] = test.Date.apply(create_month)
test['day'] = test.Date.apply(create_day)
train['Lat_int'] = train.Latitude.apply(int)
train['Long_int'] = train.Longitude.apply(int)
test['Lat_int'] = test.Latitude.apply(int)
test['Long_int'] = test.Longitude.apply(int)
# drop address columns
train = train.drop(['Address', 'AddressNumberAndStreet', 'WnvPresent', 'NumMosquitos' ], axis = 1)
test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)
# Merge with weather data
train = train.merge(weather, on = 'Date')
test = test.merge(weather, on = 'Date')
train = train.drop(['Date'], axis = 1)
test = test.drop(['Date'], axis = 1)
train.head()
weather.shape
# Convert Categorical Data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values)+list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)
train['Species']
train.head()
train['Trap'] = lbl.fit_transform(train['Trap'].values)
#test['Trap'] = lbl.fit_transform(test['Trap'].values)
train.head()
train['Trap'].unique()
test['Trap'].unique()
test['Trap'] = lbl.fit_transform(test['Trap'].values)
test.head()
test['Trap'].unique()
# Drop the columns with -1s
train = train.loc[:, (train != -1).any(axis = 0)]
test = test.loc[:, (test != -1).any(axis = 0)]
clf = ensemble.RandomForestClassifier(n_jobs = -1, n_estimators = 1000,min_samples_split = 1 )
clf.fit(train, labels)
import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

# Load dataset 
train = pd.read_csv("/kaggle/input/predict-west-nile-virus/train.csv.zip")
test = pd.read_csv("/kaggle/input/predict-west-nile-virus/test.csv.zip")
sample = pd.read_csv("/kaggle/input/predict-west-nile-virus/sampleSubmission.csv.zip")
weather = pd.read_csv("/kaggle/input/predict-west-nile-virus/weather.csv.zip")
# Get labels
labels = train.WnvPresent.values

# Not using codesum for this benchmark
weather = weather.drop('CodeSum', axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

# replace some missing values and T with -1
weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', -1)
weather = weather.replace(' T', -1)
weather = weather.replace('  T', -1)

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]

train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)
test['month'] = test.Date.apply(create_month)
test['day'] = test.Date.apply(create_day)

# Add integer latitude/longitude columns
train['Lat_int'] = train.Latitude.apply(int)
train['Long_int'] = train.Longitude.apply(int)
test['Lat_int'] = test.Latitude.apply(int)
test['Long_int'] = test.Longitude.apply(int)

# drop address columns
train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos'], axis = 1)
test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)

# Merge with weather data
train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')
train = train.drop(['Date'], axis = 1)
test = test.drop(['Date'], axis = 1)

# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)

lbl.fit(list(train['Street'].values) + list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)
test['Street'] = lbl.transform(test['Street'].values)

lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
test['Trap'] = lbl.transform(test['Trap'].values)

# drop columns with -1s
train = train.loc[:,(train != -1).any(axis=0)]
test = test.loc[:,(test != -1).any(axis=0)]

# Random Forest Classifier 
clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=1.0)
clf.fit(train, labels)

# create predictions and submission file
#predictions = clf.predict_proba(test)[:,1]
#sample['WnvPresent'] = predictions
#sample.to_csv('beat_the_benchmark.csv', index=False)
# use grid search ....