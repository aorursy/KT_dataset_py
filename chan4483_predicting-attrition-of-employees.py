#importing the required libraries

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
employee_df = pd.read_csv('../input/hrdepartment20_may_2020.csv')

employee_df.head()
employee_df.info()
employee_df.describe()
#replacing Yes/no in attrition column with 1/0

employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

employee_df.head()
employee_df.hist(figsize=(20,20))

plt.show()
#Removing EmployeeCount, EmployeeNumber, Over18, StandardHours as they dont add much value to the code



employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)
employee_df.shape
#dataframes for people who stayed or left

left_df = employee_df[employee_df['Attrition'] == 1]

stayed_df = employee_df[employee_df['Attrition'] == 0]
#based on Age

plt.figure(figsize=[15, 8])

sns.countplot(x = 'Age', hue = 'Attrition', data = employee_df)

plt.show()
plt.figure(figsize=[20,12])

plt.subplot(411)

sns.countplot(x = 'JobRole', hue = 'Attrition', data = employee_df)

plt.subplot(412)

sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = employee_df)

plt.subplot(413)

sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = employee_df)

plt.subplot(414)

sns.countplot(x = 'JobLevel', hue = 'Attrition', data = employee_df)

plt.show()


plt.figure(figsize=(10,6))



sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', shade = True, color = 'r')

sns.kdeplot(stayed_df['DistanceFromHome'], label = 'Employees who Stayed', shade = True, color = 'b')



plt.xlabel('Distance From Home')
plt.figure(figsize=(12,7))



sns.kdeplot(left_df['TotalWorkingYears'], shade = True, label = 'Employees who left', color = 'r')

sns.kdeplot(stayed_df['TotalWorkingYears'], shade = True, label = 'Employees who Stayed', color = 'b')



plt.xlabel('Total Working Years')
#checking monthly income based on job role

plt.figure(figsize=(10, 6))

sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)

employee_df.OverTime
X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]



X= X_numerical

X
y= employee_df['Attrition']

y
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(X_numerical)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



model = LogisticRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report



print("Accuracy of the Model is",100*accuracy_score(y_pred,y_test), '%')

# Testing Set Performance

ConfusionMatrix = confusion_matrix(y_pred, y_test)

sns.heatmap(ConfusionMatrix, annot=True,linewidths=1 )

plt.show()
ConfusionMatrix
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
RFConfusionMatrix = confusion_matrix(y_pred, y_test)

sns.heatmap(RFConfusionMatrix, annot=True)
print("Clasiification Report from Random Forest \n",classification_report(y_test, y_pred))