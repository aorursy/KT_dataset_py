#Call libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



HR_data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv",header =0)



HR_data.info()



HR_data.head()



# Plot for Daily Rate

sns.distplot(HR_data['DailyRate'])

plt.show()



# Plot for Education Field

sns.countplot(HR_data.EducationField)

plt.show()



# Work life balance Vs Total Working Years

sns.barplot(x = 'WorkLifeBalance', y = 'TotalWorkingYears', data = HR_data)

plt.show()



# Relationship satisfaction Vs Years in Current Role

sns.barplot(x = 'RelationshipSatisfaction', y = 'YearsInCurrentRole', data = HR_data)

plt.show()



# Training times last Year Vs Total Working Years 

sns.boxplot(HR_data['TrainingTimesLastYear'], HR_data['TotalWorkingYears'])

plt.show()



# Convert Years Since last Promotion to categorical variable

HR_data['Years_Criteria'] = pd.cut(HR_data ['YearsSinceLastPromotion'], 3, labels=['low', 'middle', 'high'])

HR_data.head()



# Joint scatter plot for Distance from home Vs Age

sns.jointplot(HR_data.DistanceFromHome,HR_data.Age, kind = "scatter")   

plt.show()



#Joint scatter plot for Total Working Years Vs Years In Current Role

sns.jointplot(HR_data.TotalWorkingYears,HR_data.YearsInCurrentRole, kind = "reg")   

plt.show()



#Joint scatter plot for Total Working Years Vs Years In Current Role

sns.jointplot(HR_data.TotalWorkingYears,HR_data.YearsInCurrentRole, kind = "reg")   

plt.show()



#Pairplot

data_hr= ['NumCompaniesWorked','YearsAtCompany','YearsWithCurrManager','YearsInCurrentRole','Attrition']

sns.pairplot(HR_data[data_hr], kind="reg", diag_kind = "kde" , hue = 'Attrition' )

plt.show()