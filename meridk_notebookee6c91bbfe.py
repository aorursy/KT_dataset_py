# unzip data

!unzip /kaggle/input/restaurant-revenue-prediction/test.csv.zip

!unzip /kaggle/input/restaurant-revenue-prediction/train.csv.zip
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# read data and drop Id column

train = pd.read_csv('./train.csv').drop('Id', axis=1)

test = pd.read_csv('./test.csv').drop('Id', axis=1)
# check for null

train.columns[train.isna().any()].tolist(), test.columns[test.isna().any()].tolist()
# change datetime format

train['Open Date'] = pd.to_datetime(train['Open Date'], format='%m/%d/%Y')   

test['Open Date'] = pd.to_datetime(test['Open Date'], format='%m/%d/%Y')



# count days from opening

train['OpenDays'] = (pd.to_datetime('now') - train['Open Date']).astype('timedelta64[D]').astype(int)

test['OpenDays'] = (pd.to_datetime('now') - test['Open Date']).astype('timedelta64[D]').astype(int)



# drop Open Date column

train.drop('Open Date', axis=1)

test.drop('Open Date', axis=1)
# count unique values of City Group

test['City Group'].value_counts(), train['City Group'].value_counts()
# change Other to 1, Big Cities to 0

test['City Group'] = test['City Group'].apply(lambda x: int(x == 'Other'))

train['City Group'] = train['City Group'].apply(lambda x: int(x == 'Big Cities'))

test.head()
# count unique values of City Group

test['Type'].value_counts(), train['Type'].value_counts()
pd.get_dummies(test['Type'])
print("ого")