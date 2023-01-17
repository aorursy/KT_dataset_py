%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Routines for linear regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
gender_submission = pd.read_csv("../input/gender_submission.csv")
train_data.head()
train_data_cp = train_data.copy()
train_data_cp["Sex"] = (train_data_cp["Sex"] == "male")*2-1
train_data_cp.corr()["Survived"]
#Distribution by Pclass

grid = GridSpec(2, 3)
labels = ["Survived","Not Survived"]
#Class 1
survived_1 = train_data_cp[(train_data_cp["Pclass"] == 1) & (train_data_cp["Survived"] == 1)].count()[0]
sizes_class_1 = [survived_1, train_data_cp[train_data_cp["Pclass"] == 1].count()[0]-survived_1]

plt.subplot(grid[0, 0], aspect=1)
plt.title("Class 1")
plt.pie(x = sizes_class_1,labels = labels,autopct='%1.1f%%')

#Class 2
survived_2 = train_data_cp[(train_data_cp["Pclass"] == 2) & (train_data_cp["Survived"] == 1)].count()[0]
sizes_class_2 = [survived_1, train_data_cp[train_data_cp["Pclass"] == 2].count()[0]-survived_2]

plt.subplot(grid[0, 2], aspect=1)
plt.title("Class 2")

plt.pie(x = sizes_class_2,labels = labels,autopct='%1.1f%%')

#Class 3
survived_3 = train_data_cp[(train_data_cp["Pclass"] == 3) & (train_data_cp["Survived"] == 1)].count()[0]
sizes_class_3 = [survived_1, train_data_cp[train_data_cp["Pclass"] == 3].count()[0]-survived_3]
plt.subplot(grid[1,1], aspect=1)
plt.title("Class 3")
plt.pie(x = sizes_class_3,labels = labels,autopct='%1.1f%%')

plt.show()
#Distribution by Sex

grid = GridSpec(1, 3)
labels = ["Survived","Not Survived"]
#Male
survived_male = train_data_cp[(train_data_cp["Sex"] == 1) & (train_data_cp["Survived"] == 1)].count()[0]
sizes_male = [survived_male, train_data_cp[train_data_cp["Sex"] == 1].count()[0]-survived_male]

plt.subplot(grid[0, 0], aspect=1)
plt.title("Male")
plt.pie(x = sizes_male,labels = labels,autopct='%1.1f%%')

#Female
survived_female = train_data_cp[(train_data_cp["Sex"] == -1) & (train_data_cp["Survived"] == 1)].count()[0]
sizes_female = [survived_female, train_data_cp[train_data_cp["Sex"] == -1].count()[0]-survived_female]

plt.subplot(grid[0, 2], aspect=1)
plt.title("Female")

plt.pie(x = sizes_female,labels = labels,autopct='%1.1f%%')


plt.show()
print("Empty training Survived feature: ",train_data["Survived"].isnull().any())
print("Empty training Pclass feature: ", train_data["Pclass"].isnull().any())
print("Empty training Sex feature: ", train_data["Sex"].isnull().any())
print("Empty training Age feature: ", train_data["Age"].isnull().any())
print("Empty testing Pclass feature: ", test_data["Pclass"].isnull().any())
print("Empty testing Sex feature: ", test_data["Sex"].isnull().any())
print("Empty testing Age feature: ", test_data["Age"].isnull().any())
mean_age = round(train_data.mean()["Age"])
train_data["Age"][train_data["Age"].isnull()] = mean_age
test_data["Age"][test_data["Age"].isnull()] = mean_age
#If it's not converted
if type(train_data["Sex"][0]) == str:
    train_data["Sex"] = (train_data["Sex"] == "male")*2-1
    test_data["Sex"] = (test_data["Sex"] == "male")*2-1
training_x = train_data[["Pclass","Sex","Age"]]
training_y = train_data["Survived"]
#Fit the model with the x points (The features) and y points (the labels (1 -Survived , 0 - Not Survived))
regr = linear_model.LinearRegression()
regr.fit(training_x,training_y)
#Predict the training values
training_y_pred = regr.predict(training_x)
print("Mean squared error: ", mean_squared_error(training_y, training_y_pred))
#Predict the test values
testing_x = test_data[["Pclass","Sex","Age"]]
testing_y_pred = regr.predict(testing_x)
print("Mean squared error: ", mean_squared_error(gender_submission["Survived"],testing_y_pred))
