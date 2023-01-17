import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data.head()
sns.countplot(data.Age > 25)

plt.show()
atr_yes = data[data['Attrition'] == 'Yes']

atr_no = data[data['Attrition'] == 'No']

plt.hist(atr_yes['Age'])
plt.figure(figsize=(12,8))

sns.barplot(x = data['Gender'], y = atr_yes['JobLevel'])
sns.barplot(x = data['JobLevel'], y = atr_yes['JobRole'])
sns.boxplot(data['Gender'], data['MonthlyIncome'])

plt.title('MonthlyIncome vs Gender Box Plot', fontsize=12)      

plt.xlabel('MonthlyIncome', fontsize=12)

plt.ylabel('Gender', fontsize=12)

plt.show()
avg_male = np.mean(data.MonthlyIncome[data.Gender == 'Male'])

avg_female = np.mean(data.MonthlyIncome[data.Gender == 'Female'])

print(avg_female/avg_male)
sns.distplot(data.MonthlyIncome[data.Gender == 'Male'], bins = np.linspace(0,20000,60))

sns.distplot(data.MonthlyIncome[data.Gender == 'Female'], bins = np.linspace(0,20000,60))

plt.legend(['Males','Females'])
plt.figure(figsize = (10,10))

plt.subplot(3,1,1)

plt.title('Sales')

sns.distplot(data.MonthlyIncome[(data.Department == 'Sales') & (data.Gender == 'Male')])

sns.distplot(data.MonthlyIncome[(data.Department == 'Sales') & (data.Gender == 'Female')])

plt.xlabel('')



plt.subplot(3,1,2)

plt.title('R&D')

sns.distplot(data.MonthlyIncome[(data.Department == 'Research & Development') & (data.Gender == 'Male')])

sns.distplot(data.MonthlyIncome[(data.Department == 'Research & Development') & (data.Gender == 'Female')])

plt.xlabel('')



plt.subplot(3,1,3)

plt.title('HR')

sns.distplot(data.MonthlyIncome[(data.Department == 'Human Resources') & (data.Gender == 'Male')])

sns.distplot(data.MonthlyIncome[(data.Department == 'Human Resources') & (data.Gender == 'Female')])
plt.figure(figsize = (10,10))

plt.subplot(2,1,1)

plt.plot(data.JobLevel,data.MonthlyIncome,'o', alpha = 0.01)

plt.xlabel('Job Level')

plt.ylabel('Monthly Income')
sns.distplot(data.TotalWorkingYears, bins = np.arange(min(data.TotalWorkingYears),max(data.TotalWorkingYears),1))

plt.ylabel('Number of Employees')
_ = plt.scatter((data['MonthlyRate'] / data['DailyRate']), data['DailyRate'])

_ = plt.xlabel('Ratio of Monthly to Daily Rate')

_ = plt.ylabel('Daily Rate')

_ = plt.title('Monthly/Daily Rate Ratio vs. Daily Rate')

plt.show()
sns.jointplot(data.MonthlyIncome ,data.Age, kind = "scatter")   

plt.show()
cont_col= ['Age', 'PerformanceRating','MonthlyIncome','Attrition']

sns.pairplot(data[cont_col], kind="reg", diag_kind = "kde" , hue = 'Attrition' )

plt.show()