import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns

from matplotlib.pyplot import figure
df=pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.isnull().values.any()
df.info()
df.describe()
df_new=df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1)
attrition=df_new.loc[df.Attrition=='Yes']

print(len(attrition))
sns.countplot(x='BusinessTravel',data=attrition)
print("Complete data")

print(df_new['BusinessTravel'].value_counts(normalize=True)*100)

print("\nBusinessTravel categorical percentage when attrition=Yes")

print(attrition['BusinessTravel'].value_counts(normalize=True)*100)
sns.countplot(x='Department',data=attrition)
df_new.JobRole.unique()
figure(figsize=(20,4)) 

sns.countplot(x='JobRole',data=attrition)
print(df.loc[df['JobRole']=='Laboratory Technician','Department'].iloc[0])

print(df.loc[df['JobRole']=='Sales Executive','Department'].iloc[0])

print(df.loc[df['JobRole']=='Research Scientist','Department'].iloc[0])
figure(figsize=(15,4))

sns.stripplot(x="DistanceFromHome", y="DailyRate", data=attrition,jitter=True)
sns.countplot(x='EnvironmentSatisfaction',data=attrition)
sns.countplot(x='Gender',data=df_new,hue='Attrition')
sns.distplot(attrition['JobInvolvement'],hist=False)
sns.lineplot(x='JobLevel',y='JobSatisfaction',data=attrition)
sns.distplot(attrition['JobLevel'],hist=False)
sns.distplot(attrition['JobSatisfaction'],hist=False)
sns.distplot(attrition['Age'],hist=False)
figure(figsize=(10,4))

overtime=attrition.loc[attrition.OverTime=='Yes']

sns.countplot(x='Age',data=overtime)
figure(figsize=(15,4))

sns.kdeplot(attrition['MonthlyRate'])

sns.rugplot(attrition['MonthlyRate'])
edjob=df_new.loc[(df.JobSatisfaction>=3) & (df.Attrition=='Yes')]

print(len(edjob))

edjob=df_new.loc[(df.JobSatisfaction<3)  & (df.Attrition=='Yes')]

print(len(edjob))
sns.countplot(x='WorkLifeBalance',data=attrition)
sns.countplot(x='MaritalStatus',data=attrition,hue='Attrition')
print("Single",len(attrition.loc[(attrition.MaritalStatus=='Single') & (attrition.OverTime=='Yes')]))

print("Married",len(attrition.loc[(attrition.MaritalStatus=='Married') & (attrition.OverTime=='Yes')]))

print("Divorced",len(attrition.loc[(attrition.MaritalStatus=='Divorced') & (attrition.OverTime=='Yes')]))
df_new[['TotalWorkingYears','YearsAtCompany']].head()
df_new['PastExperience']=df_new['TotalWorkingYears']-df_new['YearsAtCompany']
#df_new[['TotalWorkingYears','YearsAtCompany','PastExperience']]

df_new.head()
print(len(df_new.loc[(df_new.PastExperience>0) & (df_new.Attrition=='Yes')]))

print(len(df_new.loc[(df_new.PastExperience>0) & (df_new.Attrition=='No')]))
figure(figsize=(10,4))

sns.countplot(x='YearsAtCompany',data=attrition)
sns.countplot(x='YearsInCurrentRole',data=attrition)
sns.distplot(attrition['RelationshipSatisfaction'],hist=False)
sns.lineplot(y='RelationshipSatisfaction',x='YearsWithCurrManager',data=attrition)
print(len(df_new['OverTime']=='Yes'))

print(len(df_new['OverTime']=='No'))



print(len(attrition['OverTime']=='Yes'))

print(len(attrition['OverTime']=='No'))
sns.boxplot(x='OverTime',y='JobInvolvement',data=attrition)
sns.countplot(x='TrainingTimesLastYear',data=attrition)
sns.lineplot(y='JobLevel',x='Education',data=df_new)
sns.countplot(x='StockOptionLevel',data=attrition)
sns.stripplot(y='MonthlyIncome',x='Attrition',data=df_new)
figure(figsize=(10,4))

sns.countplot(x='YearsSinceLastPromotion',data=attrition)
sns.countplot(x='PerformanceRating',data=attrition)
sns.countplot(x='PercentSalaryHike',data=attrition)
df_new['JobChangeRate']=df_new['PastExperience']/df_new['NumCompaniesWorked']

m = df_new.loc[df_new['JobChangeRate'] != np.inf, 'JobChangeRate'].max()

print(m)

df_new['JobChangeRate'].replace(np.inf,m,inplace=True)

df_new['JobChangeRate']
figure(figsize=(8,4))

a=df_new.loc[df_new.Attrition=='Yes']

sns.distplot(a['JobChangeRate'],kde=False)
sns.lineplot(y='PercentSalaryHike',x='JobInvolvement',data=df_new)
sns.lineplot(y='PercentSalaryHike',x='PerformanceRating',data=df_new)