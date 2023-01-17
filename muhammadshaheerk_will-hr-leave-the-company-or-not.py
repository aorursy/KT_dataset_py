# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/general_data.csv",sep=",")
data.info()
data.columns
data.isnull().sum()
data.dropna(inplace=True)
#drop the useless columns:

df.drop(['EmployeeCount','EmployeeID','StandardHours'],axis=1, inplace = True)
# import library for corr graph (heatmap)

import seaborn as sns
corr = data.corr()
plt.figure(figsize=(16,7))
sns.heatmap(corr)
plt.show()
print(len(data[data['Attrition']=='Yes']))
print(len(data[data['Attrition']=='No']))
print("percentage of leave the company is:",(len(data[data['Attrition']=='Yes'])/len(data))*100,"%")
print("percentage of continue the company is:",(len(data[data['Attrition']=='No'])/len(data))*100,"%")
print(data['BusinessTravel'].unique())
print(data['Department'].unique())
print(data['EducationField'].unique())
print(data['Gender'].unique())
print(data['JobRole'].unique())
print(data['MaritalStatus'].unique())
print(data['Over18'].unique())
# a = data['BusinessTravel'].astype("category")
# a.value_counts()
# a.cat.codes.value_counts()
# def codei(column):
#     column.astype("category")
#     return column.cat.codes.value_counts()
from sklearn.preprocessing import LabelEncoder
lE_X = LabelEncoder()
data['BusinessTravel'] = lE_X.fit_transform(data['BusinessTravel'])
data['Department'] = lE_X.fit_transform(data['Department'])
data['EducationField'] = lE_X.fit_transform(data['EducationField'])
data['Gender'] = lE_X.fit_transform(data['Gender'])
data['JobRole'] = lE_X.fit_transform(data['JobRole'])
data['MaritalStatus'] = lE_X.fit_transform(data['MaritalStatus'])
data['Over18'] = lE_X.fit_transform(data['Over18'])
# Our dependent variable is "Attriton"

from sklearn.preprocessing import LabelEncoder
lE_Y=LabelEncoder()
data['Attrition']=lE_Y.fit_transform(data['Attrition'])
data.head()
# checking the type & nullvalues of data

data.info()
corr = data.corr()
plt.figure(figsize=(18,7))
sns.heatmap(corr)
plt.show()
y = data['Attrition']
x = data.drop('Attrition', axis = 1)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=42)
from sklearn.preprocessing import StandardScaler
Scaler_X = StandardScaler()
X_train = Scaler_X.fit_transform(X_train)
X_test = Scaler_X.transform(X_test)
#import some comman libs:
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_pred
lr.score(X_test,y_test)
#import some comman libs:
from sklearn.metrics import confusion_matrix, accuracy_score

print("Accuracy of model is:",accuracy_score(y_test,y_pred))
print()
print("Confusion matrix: \n",confusion_matrix(y_test,y_pred))