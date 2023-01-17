# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.tree import export_graphviz

import graphviz



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv") 

df.head()
df.info()
da = pd.get_dummies(df).drop(columns=['Attrition_No','OverTime_No'])

corr = da.corr()

corr.loc[:,['Attrition_Yes']].sort_values(by='Attrition_Yes', ascending=False)
from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn.metrics import accuracy_score



feature_cols = ['Age', 'DailyRate', 

       'DistanceFromHome', 'Education',       

       'HourlyRate', 'JobInvolvement', 'JobLevel', 

       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',

       'PercentSalaryHike', 'PerformanceRating',

       'RelationshipSatisfaction','EnvironmentSatisfaction','JobSatisfaction',

       'StandardHours', 'StockOptionLevel',

       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',

       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',

       'YearsWithCurrManager', 

        #'Attrition_Yes'

       'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',

       'BusinessTravel_Travel_Rarely', 'Department_Human Resources',

       'Department_Research & Development', 'Department_Sales',

       'EducationField_Human Resources', 'EducationField_Life Sciences',

       'EducationField_Marketing', 'EducationField_Medical',

       'EducationField_Other', 'EducationField_Technical Degree',

       'Gender_Male',

       'JobRole_Healthcare Representative', 'JobRole_Human Resources',

       'JobRole_Laboratory Technician', 'JobRole_Manager',

       'JobRole_Manufacturing Director', 'JobRole_Research Director',

       'JobRole_Research Scientist', 'JobRole_Sales Executive',

       'JobRole_Sales Representative', 'MaritalStatus_Divorced',

       'MaritalStatus_Married', 'MaritalStatus_Single', 'Over18_Y',

       'OverTime_Yes']

X = da[feature_cols] # Features

y = da.Attrition_Yes # Target variable



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# Import `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()



# Fit the model

rfc.fit(X_train,y_train)



# Print the results

print("Features sorted by their score:")

print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), feature_cols), reverse=True))



#Predict the response for test dataset

y_pred = rfc.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
corr.loc[:,['JobSatisfaction']].sort_values(by='JobSatisfaction', ascending=False)
from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn.metrics import accuracy_score



feature_cols = ['Age', 'DailyRate', 

       'DistanceFromHome', 'Education',       

       'HourlyRate', 'JobInvolvement', 'JobLevel', 

       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',

       'PercentSalaryHike', 'PerformanceRating',

       'RelationshipSatisfaction','EnvironmentSatisfaction',#'JobSatisfaction',

       'StandardHours', 'StockOptionLevel',

       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',

       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',

       'YearsWithCurrManager', 

       'Attrition_Yes',

       'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',

       'BusinessTravel_Travel_Rarely', 'Department_Human Resources',

       'Department_Research & Development', 'Department_Sales',

       'EducationField_Human Resources', 'EducationField_Life Sciences',

       'EducationField_Marketing', 'EducationField_Medical',

       'EducationField_Other', 'EducationField_Technical Degree',

       'Gender_Male',

       'JobRole_Healthcare Representative', 'JobRole_Human Resources',

       'JobRole_Laboratory Technician', 'JobRole_Manager',

       'JobRole_Manufacturing Director', 'JobRole_Research Director',

       'JobRole_Research Scientist', 'JobRole_Sales Executive',

       'JobRole_Sales Representative', 'MaritalStatus_Divorced',

       'MaritalStatus_Married', 'MaritalStatus_Single', 'Over18_Y',

       'OverTime_Yes']

X = da[feature_cols] # Features

y = da.JobSatisfaction # Target variable



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# Import `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()



# Fit the model

rfc.fit(X_train,y_train)



# Print the results

print("Features sorted by their score:")

print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), feature_cols), reverse=True))



#Predict the response for test dataset

y_pred = rfc.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))