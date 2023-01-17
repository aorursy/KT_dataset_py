# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_data = pd.read_csv("../input/train.csv")
print(train_data.head())
pd.set_option('display.max_columns', None)

# Any results you write to the current directory are saved as output.
print(train_data.shape)
target = train_data["Category"].unique()
print(target)
test_data = pd.read_csv("../input/test.csv")
print(test_data.head())
print(test_data.shape)
print(train_data.shape)
data_dict = {}
count = 1
for data in target:
    data_dict[data] = count
    count = count + 1
train_data["Category"] = train_data["Category"].replace(data_dict)
print(train_data)
data_week_dict= {
    "Monday": 1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6,
    "Sunday":7
}
train_data["DayOfWeek"] = train_data["DayOfWeek"].replace(data_week_dict)
test_data["DayOfWeek"] = test_data["DayOfWeek"].replace(data_week_dict)
district = train_data["PdDistrict"].unique()
data_dict_district = {}
count = 1
for data in district:
    data_dict_district[data] = count
    count+=1
train_data["PdDistrict"] = train_data["PdDistrict"].replace(data_dict_district)
test_data["PdDistrict"] = test_data["PdDistrict"].replace(data_dict_district)


print(train_data.head())
columns_train = train_data.columns
print(columns_train)
columns_test = test_data.columns
print(columns_test)
cols = columns_train.drop("Resolution")
print(cols)
train_data_new = train_data[cols]
print(train_data_new.head())
print(train_data_new.describe())
corr = train_data_new.corr()
print(corr["Category"])
skew = train_data_new.skew()
print(skew)
features = ["DayOfWeek", "PdDistrict", "X", "Y"]
X_train = train_data[features]
y_train = train_data["Category"]
X_test = test_data[features]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
from collections import OrderedDict
data_dict_new = OrderedDict(sorted(data_dict.items()))
print(data_dict_new)
#print prediction

result_dataframe = pd.DataFrame({
    "Id": test_data["Id"]
})
for key, value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count = count + 1
result_dataframe.to_csv("submission_knn.csv", index = False)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 250, criterion = 'gini',max_depth = 10)
RF.fit(X_train, y_train)
#prediction

RFpredictions = RF.predict(X_test)


#print prediction

result_dataframe = pd.DataFrame({
    "Id": test_data["Id"]
})
for key, value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in RFpredictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count = count + 1
result_dataframe.to_csv("submission_RF_v2.csv", index = False)
