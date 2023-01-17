import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
ibmhr = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
ibmhr.head()
ibmhr.shape
ibmhr.dtypes
check_null= ibmhr.isnull()
check_null.sum()
ibmHTgetNAN=ibmhr.isnull().any()
ibmHTgetNAN
ibmhr_cols=ibmhr[ibmhr.isnull().any(axis=1)]
ibmhr_cols
ibmhr.nunique()
cols = ["Over18", "StandardHours", "EmployeeCount"]
for i in cols:
    del ibmhr[i]
    
(ibmhr.select_dtypes(exclude=['int64'])).head(0)
for i in range((ibmhr.select_dtypes(exclude=['int64'])).shape[1]):
    print (ibmhr.columns.values[i],":",np.unique(ibmhr[ibmhr.columns.values[i]]),"\n")
fig, ax = plt.subplots(2,3, figsize=(10,10))             
#plt.suptitle("Distribution of various factors", fontsize=20)
sns.distplot(ibmhr['Age'],kde=False,ax=ax[0,0]) 
sns.distplot(ibmhr['MonthlyIncome'],kde=False,ax=ax[0,1]) 
sns.distplot(ibmhr['DistanceFromHome'],kde=False,ax=ax[0,2])
fig,ax = plt.subplots(3,3, figsize=(10,10))    
sns.kdeplot(ibmhr['YearsInCurrentRole'], ax = ax[0,0]) 
sns.kdeplot(ibmhr['TotalWorkingYears'], ax = ax[0,1]) 
sns.kdeplot(ibmhr['YearsAtCompany'], ax = ax[0,2])  
sns.kdeplot(ibmhr['DailyRate'], ax = ax[1,0]) 
sns.kdeplot(ibmhr['MonthlyRate'], ax = ax[1,1]) 
sns.kdeplot(ibmhr['NumCompaniesWorked'], ax = ax[1,2])  
sns.distplot(ibmhr['Age'], hist=False,ax = ax[2,0]) 
sns.distplot(ibmhr['MonthlyIncome'],hist=False, ax = ax[2,1]) 
sns.distplot(ibmhr['DistanceFromHome'],hist=False, ax = ax[2,2]) 
plt.show()
sns.kdeplot(ibmhr['WorkLifeBalance'])
sns.kdeplot(ibmhr['WorkLifeBalance'], bw=.2, label="bw: 0.8")
sns.kdeplot(ibmhr['WorkLifeBalance'], bw=2, label="bw: 4")
plt.legend();
fig,ax = plt.subplots(3,2, figsize=(20,20))  
plt.suptitle("Count Plots for various factors", fontsize=20)
sns.countplot(ibmhr['Attrition'], ax = ax[0,0]) 
sns.countplot(ibmhr['BusinessTravel'], ax = ax[0,1]) 
sns.countplot(ibmhr['Department'], ax = ax[1,0]) 
sns.countplot(ibmhr['EducationField'], ax = ax[1,1])
sns.countplot(ibmhr['Gender'], ax = ax[2,0])  
sns.countplot(ibmhr['OverTime'], ax = ax[2,1]) 
plt.xticks(rotation=20)
plt.subplots_adjust(bottom=0.4)
plt.show() 
sns.jointplot(x='MonthlyIncome', y='YearsAtCompany', data=ibmhr,kind = 'hex');
sns.jointplot(ibmhr.MonthlyIncome, ibmhr.YearsAtCompany, ibmhr, kind = 'kde');
plt.show()
sns.lmplot( x="NumCompaniesWorked", y="Age", data=ibmhr, fit_reg=False, hue='Attrition', legend=False)
plt.legend(loc='lower right')
sns.pairplot(ibmhr.iloc[:,[1,29,30,31]], hue='Attrition', size=3.5);
fig,ax = plt.subplots(2,2, figsize=(20,20))
sns.stripplot(x="Attrition", y="MonthlyIncome", data=ibmhr,ax= ax[0,0]);
sns.stripplot(x="Attrition", y="MonthlyIncome", data=ibmhr, jitter=True, ax=ax[0,1]);
sns.swarmplot(x="Department", y="MonthlyIncome",hue="Attrition" ,data=ibmhr, ax=ax[1,0]);
sns.swarmplot(x="MaritalStatus", y="MonthlyIncome",hue="Attrition" ,data=ibmhr, ax=ax[1,1]);

fig,ax = plt.subplots(figsize=(15,10))
sns.boxplot(x = 'Gender',y = 'MonthlyIncome',data=ibmhr, hue='Attrition',palette='Set3')
plt.legend(loc='best')
plt.show()
fig,ax = plt.subplots(figsize=(10,10))
sns.violinplot(x = 'Gender',y = 'MonthlyIncome',data=ibmhr, hue='Attrition',split=True,palette='Set3')
plt.legend(loc='best')
plt.show()

fig,ax = plt.subplots(2,3, figsize=(20,20))               # 'ax' has references to all the four axes
plt.suptitle("Distribution of various factors", fontsize=20)
sns.barplot(ibmhr['Gender'],ibmhr['DistanceFromHome'],hue = ibmhr['Attrition'], ax = ax[0,0]); 
sns.barplot(ibmhr['Gender'],ibmhr['YearsAtCompany'],hue = ibmhr['Attrition'], ax = ax[0,1]); 
sns.barplot(ibmhr['Gender'],ibmhr['TotalWorkingYears'],hue = ibmhr['Attrition'], ax = ax[0,2]); 
sns.barplot(ibmhr['Gender'],ibmhr['YearsInCurrentRole'],hue = ibmhr['Attrition'], ax = ax[1,0]); 
sns.barplot(ibmhr['Gender'],ibmhr['YearsSinceLastPromotion'],hue = ibmhr['Attrition'], ax = ax[1,1]); 
sns.barplot(ibmhr['Gender'],ibmhr['NumCompaniesWorked'],hue = ibmhr['Attrition'], ax = ax[1,2]); 
plt.show()
ibmhr['hike_level'] = pd.cut(ibmhr['PercentSalaryHike'], 3, labels=['Low', 'Avg', 'High']) 
sns.factorplot(x ='JobRole',y ='MonthlyIncome',hue = 'Attrition',col = 'hike_level',col_wrap=2,
               kind = 'box',
               data = ibmhr)
plt.xticks( rotation=30)
plt.show()

sns.factorplot(x ='MaritalStatus',y ='TotalWorkingYears',hue = 'Attrition',col = 'hike_level',col_wrap=2,
               data = ibmhr)
plt.xticks( rotation=30)
plt.show()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(ibmhr.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
