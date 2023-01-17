import numpy as np

import pandas as pd

import os

print(os.listdir("../input/"))
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
#data=pd.read_csv("../input/general_data.csv", sep=",")

data = pd.read_csv("../input/hr-analytics-case-study/general_data.csv",sep=",")
data.info()
data.head()
data.shape
data.describe()
print(data.columns)
#data Cleaning

data.isnull().sum()
data.isnull().any()
data.fillna(0,inplace =True)
data.isnull().any()    # No Null value
data.drop(['EmployeeCount','EmployeeID','StandardHours'],axis=1,inplace=True)
# Data Visualization 

#Find the correlation b/w all the columns



corr_cols = data[['Age','Attrition','BusinessTravel','DistanceFromHome','Education', 'EducationField','Gender', 'JobLevel', 'JobRole',

       'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',

       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',

       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']]
corr = corr_cols.corr()

plt.figure(figsize=(16,10))

sns.heatmap(corr,annot=True)

plt.show()
print(len(data))

print(len(data[data['Attrition']=='Yes']))

print(len(data[data['Attrition']=='No']))

print("percentage of yes Attrition is:",(len(data[data['Attrition']=='Yes'])/len(data))*100,"%")

print("percentage of no Attrition is:",(len(data[data['Attrition']=='No'])/len(data))*100,"%")
sns.countplot(x = "Attrition",data=data)

plt.show()
sns.countplot(x = "Attrition",data=data,hue="Gender")

plt.show()
sns.countplot(x = "Attrition",data=data,hue="JobLevel")

plt.show()
#function to creat group of ages, this helps because we have 78 differente values here

def Age(dataframe):

    dataframe.loc[dataframe['Age'] <= 30,'Age'] = 1

    dataframe.loc[(dataframe['Age'] > 30) & (dataframe['Age'] <= 40), 'Age'] = 2

    dataframe.loc[(dataframe['Age'] > 40) & (dataframe['Age'] <= 50), 'Age'] = 3

    dataframe.loc[(dataframe['Age'] > 50) & (dataframe['Age'] <= 60), 'Age'] = 4

    return dataframe



Age(data); 
sns.countplot(x = "Attrition",data=data,hue="Age")

plt.show()
#Convert all the Categorical data into numerical data

print(data['BusinessTravel'].unique())

print(data['EducationField'].unique())

print(data['Gender'].unique())

print(data['Department'].unique())

print(data['JobRole'].unique())

print(data['MaritalStatus'].unique())

print(data['Over18'].unique())



from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

data['BusinessTravel'] = labelEncoder_X.fit_transform(data['BusinessTravel'])

data['Department'] = labelEncoder_X.fit_transform(data['Department'])

data['EducationField'] = labelEncoder_X.fit_transform(data['EducationField'])

data['Gender'] = labelEncoder_X.fit_transform(data['Gender'])

data['JobRole'] = labelEncoder_X.fit_transform(data['JobRole'])

data['MaritalStatus'] = labelEncoder_X.fit_transform(data['MaritalStatus'])

data['Over18'] = labelEncoder_X.fit_transform(data['Over18'])
#Attriton is dependent var

from sklearn.preprocessing import LabelEncoder

label_encoder_y=LabelEncoder()

data['Attrition']=label_encoder_y.fit_transform(data['Attrition'])
data.head()
corr_cols = data[['Age','Attrition','BusinessTravel','DistanceFromHome','Education', 'EducationField','Gender', 'JobLevel', 'JobRole',

       'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',

       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',

       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']]
corr = corr_cols.corr()

plt.figure(figsize=(18,10))

sns.heatmap(corr, annot = True)

plt.show()
#Split data into training and Testing set:

#Choose dependent and independent var:¶

#here dependent var is Attrition and rest of the var are indepdent var.



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
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)



print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))