# Importing the necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
pd.set_option('display.max_columns',40)
df=pd.read_csv('../input/capstone-projectibm-employee-attrition-prediction/IBM HR Data new.csv')

df
df.columns
df.isnull().sum()[df.isnull().sum()!=0]
Null_values_percentage=(df.isnull().sum().sum()/len(df))*100

Null_values_percentage
### As there is only 1.5% of total null values in dataset, we will drop those null values
df=df.dropna()

df.shape
df.drop_duplicates(keep='first',inplace=True)

df.shape
df.info()
df['Age'].value_counts()
sns.distplot(df['Age'],hist=True,kde=True,color='k',bins=10)
# Majority of employees lie between the age range of 30 to 40
sns.catplot(x='Age',hue='Attrition',data=df,kind='count',height=15)
# Majority of attritions can be seen in 28 to 33 age group range
df['Attrition'].value_counts()
sns.countplot(x='Attrition',data=df,hue='Gender')
# Count of male employees are more in case of attrition
df['BusinessTravel'].value_counts()
sns.countplot(x='BusinessTravel',data=df,hue='Attrition')
sns.catplot(x='BusinessTravel',data=df,hue='Attrition',col='Department',kind='count',height=5)
# Wrt all the departments we can conclude that 'Travel_Frequently Business Travel' are in the verge towards attrition for HR Dept.
df['DailyRate'].value_counts()
sns.distplot(df['DailyRate'],bins=10,color='k')
df['DailyRate'].mean()
df['DailyRate'].min()
df['DailyRate'].max()
# The average of daily rate is somewhere around 802,minimum is 102,and maximum is 1499.
df['Department'].value_counts()
sns.countplot(df['Department'])
# Around 60% employees are working in R&D Department
sns.catplot(x='Department',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# Sales department has a high attrition rate
df['DistanceFromHome'].value_counts()
# As from info it is observed that 'Distance From Home' is object type,so we converted it to numeric type
df['DistanceFromHome']=pd.to_numeric(df['DistanceFromHome'],errors='coerce')
plt.figure(figsize=(15,10))

plt.xticks(rotation='vertical')

sns.countplot(df['DistanceFromHome'])
sns.distplot(df['DistanceFromHome'],color='k',bins=10)
# From the above count plot we can see that there are multiple instances of some numbers in int and float,so we will convert all to a single datatype
df['DistanceFromHome']=df['DistanceFromHome'].astype('int')
df['DistanceFromHome'].value_counts()
plt.figure(figsize=(15,10))

plt.xticks(rotation='vertical')

sns.countplot(df['DistanceFromHome'])
sns.distplot(df['DistanceFromHome'],color='k',bins=10)
df['DistanceFromHome'].mean()
df['DistanceFromHome'].min()
df['DistanceFromHome'].max()
# We can see that the avg distance from home is around 9Km, minimum is 1Km and maximum is 29Km.
sns.catplot(x='DistanceFromHome',hue='Attrition',col='Gender',data=df,kind='count',height=15,aspect=0.5)
# In case of both male and female,attrition rate tends to be higher when the distance exceed 10Km.
sns.catplot(x='DistanceFromHome',hue='Attrition',col='Department',data=df,kind='count',height=15,aspect=0.5)
# In case of all departments,attrition rate tends to be higher when the distance exceed 10Km.
df['Education'].value_counts()
sns.countplot(df['Education'])
# Around 30% of employees have education level of 3
sns.catplot(x='Education',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# For both male and female,attrition rate is higher for education level 1,2 and 3.
df['EducationField'].value_counts()
# As there is only 1 count in 'Test' category,so we will impute it in 'Other' category.
df.loc[df['EducationField']=='Test','EducationField']='Other'
df['EducationField'].value_counts()
plt.xticks(rotation='vertical')

sns.countplot(df['EducationField'])
# Around 70% of employees are having 'Life Sciences' and 'Medical' education field.
sns.catplot(x='EducationField',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# Attrition rate of female in 'HR' education field is less when compared to male,

# Attrition rate of female in 'Life Sciences' and 'Medical' is more when compared to male.
df['EmployeeCount'].value_counts()
df['EmployeeNumber'].value_counts()
df['Application ID'].value_counts()
df['EnvironmentSatisfaction'].value_counts()
sns.countplot(df['EnvironmentSatisfaction'])
# Count of environment satisfaction is more towards 3 and 4.
sns.catplot(x='EnvironmentSatisfaction',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# For both male and female, attrition rate is high environment satisfaction is 1 and 2. 
df['Gender'].value_counts()
sns.countplot(df['Gender'])
# Approximately female and male ratio is 3:2
sns.catplot(x='Gender',hue='Attrition',kind='count',data=df,height=5)
# For better inference, lets calculate male and female attrition rate.
df.loc[(df['Gender']=='Female') & (df['Attrition']=='Voluntary Resignation')]
Female_Attrition_Rate=1420/9283

Female_Attrition_Rate
df.loc[(df['Gender']=='Male') & (df['Attrition']=='Voluntary Resignation')]
Male_Attrition_Rate=2243/13907

Male_Attrition_Rate
# Hence, Male attrition rate is slightly higher than Female attrition rate.
df['HourlyRate'].value_counts()
# From info we can see that HourlyRate has dtype as object, so lets convert it in integer form
df.info()
df['HourlyRate']=df['HourlyRate'].astype('int')
df.info()
sns.distplot(df['HourlyRate'],color='k',bins=10)
df['HourlyRate'].mean()
df['HourlyRate'].min()
df['HourlyRate'].max()
# Avg hourly rate is around 65 and min hourly rate is 65 and max hourly rate is 100
sns.catplot(x='HourlyRate',hue='Attrition',kind='count',data=df,height=15,aspect=1)
# There is no clear evidence that HourlyRate has any impact on attrition of employees.
df['JobInvolvement'].value_counts()
sns.countplot(df['JobInvolvement'])
# Majority of employees lie in the job involvement 2 and 3
sns.catplot(x='JobInvolvement',hue='Attrition',col='Gender',data=df,kind='count')
# Job involvement 3 has slighly more attrition rate than others.
df['JobLevel'].value_counts()
sns.countplot(df['JobLevel'])
# Majority of employees lie in the job level 1 and 2
sns.catplot(x='JobLevel',hue='Attrition',col='Gender',data=df,kind='count')
# Attrition rate is higher in job level 1 and 2.
df['JobRole'].value_counts()
plt.xticks(rotation='vertical')

sns.countplot(df['JobRole'])
# Count of employees is more in job role as Sales Executive,Laboratory Technician,Research Scientist.
g=sns.catplot(x='JobRole',hue='Attrition',col='Gender',data=df,kind='count',height=7)

g.set_xticklabels(rotation=90)
# Job role as Sales Representative has the highest attrition rate for both male and female,

# Job role as HR has high rate of attrition in case of female gender.
df['JobSatisfaction'].value_counts()
sns.countplot(df['JobSatisfaction'])
# Job Satisfaction count for 3 and 4 are more than 1 and 2.
sns.catplot(x='JobSatisfaction',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Higher attrition rate can be seen in Job Satisfaction level 1 and 2.
df['MaritalStatus'].value_counts()
sns.countplot(df['MaritalStatus'])
# Count of married employees is more
sns.catplot(x='MaritalStatus',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Attrition rate in singles are higher for both male and female
df['MonthlyIncome'].value_counts()
# As,monthly income column has object dtype, we need to convert it in integer form.
df['MonthlyIncome']=df['MonthlyIncome'].astype('int')
sns.distplot(df['MonthlyIncome'],bins=10,color='k')
df['MonthlyIncome'].mean()
df['MonthlyIncome'].min()
df['MonthlyIncome'].max()
# Minimum monthly income of employees is 1009 and maximum monthly income of employees is 19999 and avg monthly income of employees is 6507.

# Majority of employees are having monthly income lower than 5000.
df['MonthlyRate'].value_counts()
sns.distplot(df['MonthlyRate'],20,color='k')
df['MonthlyRate'].mean()
df['MonthlyRate'].min()
df['MonthlyRate'].max()
# Avg monthly rate of employees is around 14302,min monthly rate is 2094 and max monthly rate is 26999.
df['NumCompaniesWorked'].value_counts()
sns.countplot(df['NumCompaniesWorked'])
# Maximum employees have worked in only 1 company.
sns.catplot(x='NumCompaniesWorked',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# It can be observed that employees who have worked in 1 company have higher attrition rate
df['Over18'].value_counts()
df['OverTime'].value_counts()
sns.countplot(df['OverTime'])
# Approximately ratio of employees doing overtime and employees not doing overtime is 30:70
sns.catplot(x='OverTime',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# A very high attrition rate is seen in employees who are doing overtime for both male and female.
sns.catplot(x='OverTime',hue='Gender',data=df,kind='count',height=7)
# Male has a higher attrition rate in both cases
df['PercentSalaryHike'].value_counts()
sns.countplot(df['PercentSalaryHike'])
# Majority of employees got a salary hike less than 15%
sns.catplot(x='PercentSalaryHike',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Higher attrition is observed in cases where the salary hike is less than 16% for male when compared to female.
df['PerformanceRating'].value_counts()
sns.countplot(df['PerformanceRating'])
# There are very few employees who have performance rating 4.
sns.catplot(x='PerformanceRating',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Performance Rating 3 has higher rate of attrition for both male and female.
df['RelationshipSatisfaction'].value_counts()
sns.countplot(df['RelationshipSatisfaction'])
# Count of employees having relationship satisfaction 3,4 are more than 1,2.
sns.catplot(x='RelationshipSatisfaction',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Higher attrition is observed in lower relationship satisfaction for both genders
df['StandardHours'].value_counts()
df['StockOptionLevel'].value_counts()
sns.countplot(df['StockOptionLevel'])
# There are many employees who does not have stock options level,

# As the stock options level increases the count of employees reduces.
sns.catplot(x='StockOptionLevel',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Higher attrition rate is observed in lower stock options level for both genders.
df['TotalWorkingYears'].value_counts()
sns.distplot(df['TotalWorkingYears'],bins=10,color='k')
plt.figure(figsize=(10,10))

plt.xticks(rotation='vertical')

sns.countplot(df['TotalWorkingYears'])
# Maximum number of employees have total working years as 10 and the count decreases gradually after 10 years.
sns.catplot(x='TotalWorkingYears',hue='Attrition',data=df,kind='count',height=15)
# Higher attrition rate is observed for employees having total working years less than 10 years.
df['TrainingTimesLastYear'].value_counts()
sns.countplot(df['TrainingTimesLastYear'])
# Maximum employees where trained 2 to 3 times since last year
sns.catplot(x='TrainingTimesLastYear',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# Higher attrition rate can be seen where number of trainings given to employees are less for both gender.
df['WorkLifeBalance'].value_counts()
sns.countplot(df['WorkLifeBalance'])
# Count of employees having worklife balance as 3 is more wrt others
sns.catplot(x='WorkLifeBalance',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# Lower work life balance has somewhat high rate of attrition
sns.catplot(x='WorkLifeBalance',hue='Attrition',col='Department',data=df,kind='count',height=7)
# HR Department has less attrition rate in any cases of work life balance
df['YearsAtCompany'].value_counts()
sns.distplot(df['YearsAtCompany'],bins=20,color='k')
# Count of employees is maximum who have worked less than 8 years
sns.catplot(x='YearsAtCompany',hue='Attrition',data=df,kind='count',height=15)
# We can see higher attrition rate for those employees who have worked for less than 10 years
df['YearsInCurrentRole'].value_counts()
sns.distplot(df['YearsInCurrentRole'],bins=20,color='k')
# Count of employees having 2 to 3 years in current role are more.
sns.catplot(x='YearsInCurrentRole',hue='Attrition',data=df,kind='count',height=10)
# After 5 years in same role,attrition rate gradually decreases with increase in years.
df.info()
df['YearsSinceLastPromotion'].value_counts()
sns.distplot(df['YearsSinceLastPromotion'],bins=20,color='k')
sns.countplot(df['YearsSinceLastPromotion'])
# Majority of employees are in the category of having 0,1 or 2 years since last promotion.
sns.catplot(x='YearsSinceLastPromotion',hue='Attrition',data=df,kind='count',height=10)
# Attrition rate is higher where Years since last promotion is less than 7
df['YearsWithCurrManager'].value_counts()
sns.distplot(df['YearsWithCurrManager'],bins=20,color='k')
plt.figure(figsize=(10,7))

plt.xticks(rotation='vertical')

sns.countplot(df['YearsWithCurrManager'])
# Majority of employees areworking with their manager for around 2 years.
sns.catplot(x='YearsWithCurrManager',hue='Attrition',data=df,kind='count',height=10)
# As the employees work for more years with same manager,they get mentally attached with that manager and have a good comfort zone.

# Hence, they get retained for a longer period of time.

# But there are a few exceptions where the attrition rate is high even if the years are more.This maybe due to internal disputes.So,regular counselling should be done.
df['Employee Source'].value_counts()
# Since there is only 1 entry in Test,we will simply shift in other group
df.loc[df['Employee Source']=='Test','Employee Source']='Company Website'
df['Employee Source'].value_counts()
plt.xticks(rotation='vertical')

sns.countplot(df['Employee Source'])
# Around 25% employee source is Company Website, so we should management to emhance its worth more.
sns.catplot(x='Employee Source',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# At the same time,it is observed that the maximum attrition is taking place for those employees who have joined organization through companies website.

# Hence, reality check should be done in the website.
df.head()
df.shape
df.info()
# Dropping the unnecessary columns
df1=df.drop(['EmployeeCount','EmployeeNumber','Application ID','StandardHours','Over18'],axis=1)
df1.shape
df1.head()
df1['Attrition']=df1['Attrition'].apply(lambda x:1 if x=='Voluntary Resignation' else 0)
plt.figure(figsize=(35,15))

sns.heatmap(df1.corr(),cmap='rainbow',mask=abs(df1.corr())<0.05,annot=True)
plt.figure(figsize=(25,25))

sns.heatmap(df1.corr(),cmap='gnuplot2',mask=abs(df1.corr())<0.05,annot=True)
plt.figure(figsize=(25,25))

sns.heatmap(df1.corr(),cmap='gnuplot2',mask=abs(df1.corr())<0.05,annot=True)
plt.figure(figsize=(25,25))

sns.heatmap(df1.corr(),cmap='cubehelix',mask=abs(df1.corr())<0.1,annot=True)
plt.figure(figsize=(25,25))

sns.heatmap(df1.corr(),cmap='gist_rainbow',mask=abs(df1.corr())<0.2,annot=True)
plt.figure(figsize=(25,25))

sns.heatmap(df1.corr(),cmap='cubehelix',mask=abs(df1.corr())<0.3,annot=True)
plt.figure(figsize=(25,25))

sns.heatmap(df1.corr(),cmap='ocean',mask=abs(df1.corr())<0.4,annot=True)
plt.figure(figsize=(25,25))

sns.heatmap(df1.corr(),cmap='gnuplot',mask=abs(df1.corr())<0.5,annot=True)
plt.figure(figsize=(25,25))

sns.heatmap(df1.corr(),cmap='terrain',mask=abs(df1.corr())<0.6,annot=True)
# Impact of Age on Attrition of employees

sns.catplot(x='Age',hue='Attrition',data=df,kind='count',height=15)
# Impact of Job Level on Attrition of employees

sns.catplot(x='JobLevel',hue='Attrition',data=df,kind='count')
# Impact of Marital Status on Attrition of employees

sns.catplot(x='MaritalStatus',hue='Attrition',data=df,kind='count',height=7)
# Monthly Income affecting Attrition rate:

sns.barplot(x='Attrition',y='MonthlyIncome',data=df)
sns.relplot(x='JobInvolvement',y='MonthlyIncome',hue='Attrition',data=df,size='MonthlyIncome')
# Business Travel affecting attrition rate

sns.countplot(x='BusinessTravel',hue='Attrition',data=df)
df1.info()
df1['Attrition'].value_counts()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df1['BusinessTravel'].value_counts()
# We will do get dummies or ohe for this column
df1['Gender'].value_counts()
df1['Gender']=le.fit_transform(df1['Gender'])
df1['JobRole'].value_counts()
# We will do get dummies or ohe for this column
df1['JobSatisfaction'].value_counts()
df1['JobSatisfaction']=df1['JobSatisfaction'].astype('int')
df1.info()
df1['MaritalStatus'].value_counts()
# We will do get dummies or ohe for this column
df1['OverTime'].value_counts()
df1['OverTime']=le.fit_transform(df1['OverTime'])
df1['PercentSalaryHike'].value_counts()
df1['PercentSalaryHike']=df1['PercentSalaryHike'].astype('int')
df1['Employee Source'].value_counts()
# We will do get dummies or ohe for this column
df1.info()
df1=pd.get_dummies(df1,drop_first=True)
df1.head()
plt.figure(figsize=(50,35))

sns.heatmap(df1.corr(),annot=True,mask=abs(df1.corr())<0.05)
# Now our dataset is cleaned and ready for processing