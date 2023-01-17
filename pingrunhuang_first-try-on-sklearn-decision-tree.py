# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.describe()
import matplotlib as plot
from sklearn.model_selection import train_test_split
# drop the unnecessary columns
data=data.drop(columns=["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"])
print(data.dtypes)
categorial_columns=["Attrition", "BusinessTravel", "Department", "EducationField", "Gender", 
                   "JobRole", "MaritalStatus", "OverTime"]
from sklearn.preprocessing import LabelEncoder
for column in categorial_columns:
#     copy_data=data.copy()
    labelEncoder = LabelEncoder()
    data[column]=labelEncoder.fit_transform(data[column])
data.head()

# # converting numerical factor into factor
# data["Education"]=data["Education"].astype("category")
# data["EnvironmentSatisfaction"]=data["EnvironmentSatisfaction"].astype("category")
# data["JobLevel"]=data["JobLevel"].astype("category")
# data["JobSatisfaction"]=data["JobSatisfaction"].astype("category")
# train test splitting
train_data, test_data = train_test_split(data, test_size=0.3)
train_y=train_data["Attrition"]
train_x=train_data.drop("Attrition",axis=1)
test_y=test_data["Attrition"]
test_x=test_data.drop("Attrition",axis=1)
from sklearn import tree
cls = tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)
cls.fit(X=train_x.as_matrix(), y=train_y)
tree.export_graphviz(cls, out_file="tree.dot")
import subprocess
command = ["dot", "-Tpng", "tree.dot", "-o", "tree.png"]
subprocess.check_call(command)
from sklearn import metrics
y_predict = cls.predict(test_x)
print("accuracy:", metrics.accuracy_score(test_y, y_predict))
print("classfication report:", metrics.classification_report(test_y, y_predict))
print("confusion matrix:", metrics.confusion_matrix(test_y, y_predict))
