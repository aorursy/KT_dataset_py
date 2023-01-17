# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
train_data.head()
test_data = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip')
test_data.head()
ss = pd.read_csv('/kaggle/input/sf-crime/sampleSubmission.csv.zip')
ss.head()
train_data.columns
train_data['Category'].value_counts()
test_data.columns
train_data.shape
test_data.shape
train_data.info()
test_data.info()
train_data['Address'].value_counts()
test_data['Address'].value_counts()
train_data.isna().sum()
test_data.isna().sum()
target = train_data['Category'].unique()
target
data_dict = {}
count = 1
for data in target:
    data_dict[data] = count
    count += 1
train_data['Category'] = train_data['Category'].replace(data_dict)

data_week_dict = {
    'Monday' : 1,
    'Tuesday' : 2,
    'Wednesday' : 3,
    'Thursday' : 4,
    'Friday' : 5,
    'Saturday' : 6,
    'Sunday' : 7
}

train_data['DayOfWeek'] = train_data['DayOfWeek'].replace(data_week_dict)
test_data['DayOfWeek'] = test_data['DayOfWeek'].replace(data_week_dict)
district = train_data['PdDistrict'].unique()
data_dict_district = {}
count = 1
for data in district:
    data_dict_district[data] = count
    count += 1
    
train_data['PdDistrict'] = train_data['PdDistrict'].replace(data_dict_district)
test_data['PdDistrict'] = test_data['PdDistrict'].replace(data_dict_district)
print(train_data.head())
columns_train = train_data.columns
columns_test = test_data.columns
cols = columns_train.drop('Resolution')
print(cols)
train_data_new = train_data[cols]
print(train_data_new.head())
features = ['DayOfWeek', 'PdDistrict', 'X', 'Y']
X_train = train_data[features]
X_test = test_data[features]
y_train = train_data['Category']
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)
predictions = log.predict(X_test)

for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_logistic.csv", index=False) 
