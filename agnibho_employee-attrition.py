import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns

import os
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head(15).T
data.dtypes
data.describe().T
data['Attrition'].value_counts()
data.isnull().sum()
cor = data.corr()
cor
data.shape
data['Over18'].value_counts()
data['StandardHours'].value_counts()
data['EmployeeCount'].value_counts()
data = data.drop(['EmployeeCount','StandardHours','Over18'],axis=1)
data.shape
plt.boxplot(data['Age'])

plt.show()
plt.hist(data['Age'])

plt.show()
data['Age'].value_counts()
data['Attrition'].value_counts()
data.head()
data['BusinessTravel'].value_counts()
plt.bar(data['BusinessTravel'],height= 1500)

plt.show()
plt.boxplot(data['DailyRate'])

plt.show()
plt.hist(data['DailyRate'])

plt.show()
plt.hist(data['DailyRate'])

plt.yscale('log')

plt.show()
data['Department'].value_counts()
plt.bar(data['Department'],height=1000)
data['DistanceFromHome'].value_counts()
plt.boxplot(data['DistanceFromHome'])

plt.show()
plt.hist(data['DistanceFromHome'])

plt.show()
data['Education'].value_counts()
data['EducationField'].value_counts()
data['EmployeeNumber'].value_counts()
data = data.drop(['EmployeeNumber'],axis=1)
data.head().T
plt.boxplot(data['HourlyRate'])

plt.show()
plt.figure(1)

plt.subplot(1,3,1)

plt.hist(data['HourlyRate'])



plt.subplot(1,3,2)

plt.hist(data['HourlyRate'])

plt.yscale('log')





plt.subplot(1,3,3)

plt.hist(np.log(data['HourlyRate']))

plt.show()
data['JobRole'].value_counts()
plt.boxplot(data['MonthlyIncome'])

plt.show()
data['MonthlyIncome'].describe()
plt.hist(data['MonthlyIncome'])

plt.show()
iqr = ((np.percentile(data['MonthlyIncome'],75))- (np.percentile(data['MonthlyIncome'],25)))

limit = ((np.percentile(data['MonthlyIncome'],75)) + (1.5*iqr))

data.loc[data['MonthlyIncome']>limit,'MonthlyIncome'] = np.median(data['MonthlyIncome'])
plt.boxplot(data['MonthlyRate'])

plt.show()
data['NumCompaniesWorked'].value_counts()
plt.hist(data['NumCompaniesWorked'])

plt.show()
data['OverTime'].value_counts()
data['PercentSalaryHike'].value_counts()
plt.boxplot(data['PercentSalaryHike'])

plt.show()
plt.hist(data['PercentSalaryHike'])

plt.show()
plt.hist(np.log(data['PercentSalaryHike']))

plt.yscale('log')

plt.show()
data['PerformanceRating'].value_counts()
data['RelationshipSatisfaction'].value_counts()
data['StockOptionLevel'].value_counts()
plt.hist(data['StockOptionLevel'])

plt.show()
data['TotalWorkingYears'].value_counts()
plt.hist(data['TotalWorkingYears'])

plt.show()
data['TrainingTimesLastYear'].value_counts()
data['WorkLifeBalance'].value_counts()
data['YearsAtCompany'].value_counts()
data['YearsInCurrentRole'].value_counts()
plt.hist(data['YearsSinceLastPromotion'])

plt.show()
data['YearsSinceLastPromotion'].value_counts()
data['YearsWithCurrManager'].value_counts()
data.head().T
data.dtypes
data.loc[data['Attrition']=='No','Attrition']=0

data.loc[data['Attrition']=='Yes','Attrition']=1
data.loc[data['BusinessTravel']=='Non-Travel','BusinessTravel']=0

data.loc[data['BusinessTravel']=='Travel_Rarely','BusinessTravel']=1

data.loc[data['BusinessTravel']=='Travel_Frequently','BusinessTravel']=2
data.loc[data['OverTime']=='No','OverTime']=0

data.loc[data['OverTime']=='Yes','OverTime']=1
data.dtypes
data = pd.get_dummies(data)
data.head(8).T
data = data.drop(['Department_Human Resources','EducationField_Life Sciences','Gender_Female','JobRole_Laboratory Technician','MaritalStatus_Divorced'],axis=1)
data.dtypes
data.shape
features = list((data.drop(['Attrition'],axis=1)).columns)

target = 'Attrition'

print(features)

print(target)

print(len(features))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data[features],data[target],test_size=0.3,random_state=1)
print("X_train:",len(x_train))

print("X_test:",len(x_test))

print("Y_train:",len(y_train))

print("Y_test:",len(y_test))
y_test.value_counts()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=35,criterion="entropy")

rf.fit(x_train,y_train)
from sklearn import metrics

print("Random Forest")

print("Accuracy: ",rf.score(x_test,y_test))

y_pred = rf.predict(x_test)

print("Precision: ",metrics.precision_score(y_test,y_pred))

print("Recall: ",metrics.recall_score(y_test,y_pred))

print("Confusion Matrix: \n",metrics.confusion_matrix(y_test,y_pred))
majority_class = data[data['Attrition']==0]

minority_class = data[data['Attrition']==1]

print(len(majority_class))

print(len(minority_class))
from sklearn.utils import resample

minority_class_upsampled = resample(minority_class,replace=True,n_samples=1233,random_state=1)
data_balanced = pd.concat([majority_class,minority_class_upsampled])
data_balanced['Attrition'].value_counts()
x_train1,x_test1,y_train1,y_test1 = train_test_split(data_balanced[features],data_balanced[target],test_size=0.3,random_state = 1)
print('RandomForest')

rf.fit(x_train1,y_train1)
y_test1.value_counts()
print("Random Forest")

print("Accuracy: ",rf.score(x_test1,y_test1))

y_pred1 = rf.predict(x_test1)

print("Precision: ",metrics.precision_score(y_test1,y_pred1))

print("Recall: ",metrics.recall_score(y_test1,y_pred1))

print("Confusion Matrix: \n",metrics.confusion_matrix(y_test1,y_pred1))
feature_importances = pd.DataFrame(rf.feature_importances_,

                                   index = features,

                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances