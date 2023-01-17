import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline
hr = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv", header =0)

hr.head()
hr.shape
hr.dtypes
hr[hr.isnull().any(axis=1)]
hr.nunique()
cols = ["Over18", "StandardHours", "EmployeeCount"]

for i in cols:

    del hr[i]
hr.nunique()
(hr.select_dtypes(exclude=['int64'])).head(0)
for i in range((hr.select_dtypes(exclude=['int64'])).shape[1]):

    print (hr.columns.values[i],":",np.unique(hr[hr.columns.values[i]]),"\n")
fig,ax = plt.subplots(2,3, figsize=(10,10))               # 'ax' has references to all the four axes

plt.suptitle("Distribution of various factors", fontsize=20)

sns.distplot(hr['Age'], ax = ax[0,0]) 

sns.distplot(hr['MonthlyIncome'], ax = ax[0,1]) 

sns.distplot(hr['DistanceFromHome'], ax = ax[0,2]) 

sns.kdeplot(hr['YearsInCurrentRole'], ax = ax[1,0]) 

sns.kdeplot(hr['TotalWorkingYears'], ax = ax[1,1]) 

sns.kdeplot(hr['YearsAtCompany'], ax = ax[1,2])  

plt.show()
sns.kdeplot(hr['WorkLifeBalance'])

sns.kdeplot(hr['WorkLifeBalance'], bw=.2, label="bw: 0.8")

sns.kdeplot(hr['WorkLifeBalance'], bw=2, label="bw: 4")

plt.legend();
fig,ax = plt.subplots(2,3, figsize=(20,20))               # 'ax' has references to all the four axes

plt.suptitle("Distribution of various factors", fontsize=20)

sns.countplot(hr['Attrition'], ax = ax[0,0]) 

sns.countplot(hr['BusinessTravel'], ax = ax[0,1]) 

sns.countplot(hr['Department'], ax = ax[0,2]) 

sns.countplot(hr['EducationField'], ax = ax[1,0])

sns.countplot(hr['Gender'], ax = ax[1,1])  

sns.countplot(hr['OverTime'], ax = ax[1,2]) 

plt.xticks(rotation=20)

plt.subplots_adjust(bottom=0.4)

plt.show()
sns.jointplot(x='MonthlyIncome', y='YearsAtCompany', data=hr,kind = 'hex');

plt.show()
sns.jointplot(hr.MonthlyIncome, hr.YearsAtCompany, hr, kind = 'kde');
sns.pairplot(hr.iloc[:,[1,29,30,31]], hue='Attrition', size=3.5);
fig,ax = plt.subplots(2,2, figsize=(20,20))

sns.stripplot(x="Attrition", y="MonthlyIncome", data=hr,ax= ax[0,0]);

sns.stripplot(x="Attrition", y="MonthlyIncome", data=hr, jitter=True, ax=ax[0,1]);

sns.swarmplot(x="Department", y="MonthlyIncome",hue="Attrition" ,data=hr, ax=ax[1,0]);

sns.swarmplot(x="MaritalStatus", y="MonthlyIncome",hue="Attrition" ,data=hr, ax=ax[1,1]);
fig,ax = plt.subplots(figsize=(15,10))

sns.boxplot(x = 'Gender',y = 'MonthlyIncome',data=hr, hue='Attrition',palette='Set3')

plt.legend(loc='best')

plt.show()
fig,ax = plt.subplots(figsize=(10,10))

sns.violinplot(x = 'Gender',y = 'MonthlyIncome',data=hr, hue='Attrition',split=True,palette='Set3')

plt.legend(loc='best')

plt.show()
fig,ax = plt.subplots(2,3, figsize=(20,20))               # 'ax' has references to all the four axes

plt.suptitle("Distribution of various factors", fontsize=20)

sns.barplot(hr['Gender'],hr['DistanceFromHome'],hue = hr['Attrition'], ax = ax[0,0]); 

sns.barplot(hr['Gender'],hr['YearsAtCompany'],hue = hr['Attrition'], ax = ax[0,1]); 

sns.barplot(hr['Gender'],hr['TotalWorkingYears'],hue = hr['Attrition'], ax = ax[0,2]); 

sns.barplot(hr['Gender'],hr['YearsInCurrentRole'],hue = hr['Attrition'], ax = ax[1,0]); 

sns.barplot(hr['Gender'],hr['YearsSinceLastPromotion'],hue = hr['Attrition'], ax = ax[1,1]); 

sns.barplot(hr['Gender'],hr['NumCompaniesWorked'],hue = hr['Attrition'], ax = ax[1,2]); 

plt.show()
hr['hike_level'] = pd.cut(hr['PercentSalaryHike'], 3, labels=['Low', 'Avg', 'High']) 

sns.factorplot(x ='JobRole',y ='MonthlyIncome',hue = 'Attrition',col = 'hike_level',col_wrap=2,

               kind = 'box',

               data = hr)

plt.xticks( rotation=30)

plt.show()
sns.factorplot(x ='MaritalStatus',y ='TotalWorkingYears',hue = 'Attrition',col = 'hike_level',col_wrap=2,

               data = hr)

plt.xticks( rotation=30)

plt.show()
#Plot a correlation map for all numeric variables

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(hr.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()