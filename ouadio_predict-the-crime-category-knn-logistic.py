#Load the important libraries neede

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.



train_data = pd.read_csv("../input/train.csv")



# Display the first few rows of the data

print(train_data.head())



# Get the shape of the data just to know how many rows and columns it contains

print(train_data.shape)



# Any results you write to the current directory are saved as output.
#target variable



target = train_data["Category"].unique()

print(target)



# There are multiple categorical values. It looks like a multi class classification problem.
#Let's read the test data now



test_data = pd.read_csv("../input/test.csv")

print(test_data.head())

print(test_data.shape)



#Test data does not have the target variable and the resolution
data_dict = {}

count = 1

for data in target:

    data_dict[data] = count

    count+=1

train_data["Category"] = train_data["Category"].replace(data_dict)



#Replacing the day of weeks

data_week_dict = {

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

#District

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



# All the numeric columns have no missing values.
corr = train_data_new.corr()

print(corr["Category"])

 

# There is no strong correlation of category with any numeric value
#Calculate the skew



skew = train_data_new.skew()

print(skew)
#Let's use knn algorithm on numeric columns



features = ["DayOfWeek", "PdDistrict",  "X", "Y"]

X_train = train_data[features]

y_train = train_data["Category"]

X_test = test_data[features]



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
from collections import OrderedDict

data_dict_new = OrderedDict(sorted(data_dict.items()))

print(data_dict_new)

                
#print(type(predictions))

result_dataframe = pd.DataFrame({

    "Id": test_data["Id"]

})

for key,value in data_dict_new.items():

    result_dataframe[key] = 0

count = 0

for item in predictions:

    for key,value in data_dict.items():

        if(value == item):

            result_dataframe[key][count] = 1

    count+=1

result_dataframe.to_csv("submission_knn.csv", index=False) 
#Logistic Regression
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression()

lgr.fit(X_train, y_train)

predictions = knn.predict(X_test)



#print(type(predictions))

result_dataframe = pd.DataFrame({

    "Id": test_data["Id"]

})

for key,value in data_dict_new.items():

    result_dataframe[key] = 0

count = 0

for item in predictions:

    for key,value in data_dict.items():

        if(value == item):

            result_dataframe[key][count] = 1

    count+=1

result_dataframe.to_csv("submission_logistic.csv", index=False) 

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