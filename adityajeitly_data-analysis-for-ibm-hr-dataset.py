# Call libraries neded to run the dataset

import numpy as np 

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns

import warnings    
## Read CSV file into variable

data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data.head()
# Show plot for daily rate

sns.distplot(data['DailyRate'])

plt.show()
# Show plot for Education Field

sns.countplot(data.EducationField)

plt.show()
# How work life balance changes with years at the company

sns.barplot(x = 'WorkLifeBalance', y = 'TotalWorkingYears', data = data)

plt.show()
# How years in current role relates with relationship satisfaction

sns.barplot(x = 'RelationshipSatisfaction', y = 'YearsInCurrentRole', data = data)

plt.show()
# How training times last Year corresponds with total working hours 

sns.boxplot(data['TrainingTimesLastYear'], data['TotalWorkingYears'])

plt.show()
# We will create another discrete variable by cutting Years since last promotion into three parts

data['Years_Criteria'] = pd.cut(data ['YearsSinceLastPromotion'], 3, labels=['low', 'middle', 'high'])

data.head()
## Correlation between Age and Distance from home

sns.jointplot(data.DistanceFromHome,data.Age, kind = "scatter")   

plt.show()
#Scatter Plot to show Total Working years and Years in Current role

sns.jointplot(data.TotalWorkingYears,data.YearsInCurrentRole, kind = "reg")   

plt.show()
## Joint plots with hex bins for Worklife balance and Working Years

sns.jointplot(data.WorkLifeBalance,data.TotalWorkingYears, kind = "hex") 

plt.show()
data_hr= ['NumCompaniesWorked','YearsAtCompany','YearsWithCurrManager','YearsInCurrentRole','Attrition']

sns.pairplot(data[data_hr], kind="reg", diag_kind = "kde" , hue = 'Attrition' )

plt.show()