##Importing the packages

#Data processing packages

import numpy as np 

import pandas as pd 



#Visualization packages

import matplotlib.pyplot as plt 

import seaborn as sns 



import warnings

warnings.filterwarnings('ignore')
#Import Employee Attrition data

data=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
#Find the size of the data Rows x Columns

data.shape
#Display first 5 rows of Employee Attrition data

data.head()
#Find Basic Statistics like count, mean, standard deviation, min, max etc.

data.describe()
#Find the the information about the fields, field datatypes and Null values

data.info()
cat_cols = data.columns[data.dtypes=='object']

data_cat = data[cat_cols]

print(cat_cols)

print(cat_cols.shape)

data_cat.head()
#A lambda function is a small anonymous function.

#A lambda function can take any number of arguments, but can only have one expression.

data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)
data.head()
data[data.Attrition == 1].head()
num_cols = data.columns[data.dtypes!='object']

data_num = data[num_cols]

print(num_cols)

print(num_cols.shape)

data_num.head()
data.corrwith(data.Attrition, axis = 0).sort_values().head()
data.corrwith(data.Attrition, axis = 0).sort_values(ascending = False).head()
sns.countplot(data.TotalWorkingYears, hue=data.Attrition)
sns.countplot(data.DistanceFromHome, hue=data.Attrition)
data.JobLevel.value_counts().plot.bar()
sns.countplot(data.JobLevel, hue=data.Attrition)
data[data.Attrition==1].JobLevel.value_counts(normalize=True, sort=False).plot.bar()
data[data.Attrition==1].DistanceFromHome.value_counts(normalize=True, sort=False).plot.bar()
plt.figure(figsize=(20, 20)) ; sns.heatmap(data_num.corr(), annot=True)
g = sns.pairplot(data_num.loc[:,'Age':'DistanceFromHome']); g.fig.set_size_inches(15,15)

#data_num.loc[:,'Age':'DistanceFromHome']
g = sns.pairplot(data_num.loc[:,'Education':'HourlyRate']); g.fig.set_size_inches(15,15)
g = sns.pairplot(data_num.loc[:,'JobInvolvement':'MontlyRate']); g.fig.set_size_inches(15,15)
g = sns.pairplot(data_num.loc[:,'NumCompaniesWorked':'StandardHours']); g.fig.set_size_inches(15,15)
g = sns.pairplot(data_num.loc[:,'StockOptionLevel':'YearsAtCompany']); g.fig.set_size_inches(15,15)
g = sns.pairplot(data_num.loc[:,'YearsInCurrentRole':'YearsWithCurrManager']); g.fig.set_size_inches(15,15)
g = sns.pairplot(data_num); g.fig.set_size_inches(15,15)
data_num.hist(layout = (9, 3), figsize=(24, 48), color='blue', grid=False, bins=15)
#Find attrition size (Values)

data['Attrition'].value_counts()
pd.crosstab(data.BusinessTravel, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.Department, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.Education, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.EducationField, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.EnvironmentSatisfaction, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.Gender, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.JobInvolvement, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.JobLevel, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.JobRole, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.JobSatisfaction, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.MaritalStatus, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.OverTime, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.PerformanceRating, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.RelationshipSatisfaction, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.StockOptionLevel, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
pd.crosstab(data.WorkLifeBalance, data.Attrition, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')
#A lambda function can take any number of arguments, but can only have one expression.

#Change the Attrition from Yes/No to binary 1/0

data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)
#Comparing the numeric fields agains Attrition using boxplots

plt.figure(figsize=(24,12))

plt.subplot(231)  ; sns.boxplot(x='Attrition',y='Age',data=data)

plt.subplot(232)  ; sns.boxplot(x='Attrition',y='DailyRate',data=data)

plt.subplot(233)  ; sns.boxplot(x='Attrition',y='DistanceFromHome',data=data)

plt.subplot(234)  ; sns.boxplot(x='Attrition',y='HourlyRate',data=data)

plt.subplot(235)  ; sns.boxplot(x='Attrition',y='MonthlyIncome',data=data)

plt.subplot(236)  ; sns.boxplot(x='Attrition',y='PercentSalaryHike',data=data)
#Comparing the numeric fields agains Attrition using boxplots

plt.figure(figsize=(24,12))

plt.subplot(231)  ; sns.boxplot(x='Attrition',y='MonthlyRate',data=data)

plt.subplot(232)  ; sns.boxplot(x='Attrition',y='NumCompaniesWorked',data=data)

plt.subplot(233)  ; sns.boxplot(x='Attrition',y='TotalWorkingYears',data=data)

plt.subplot(234)  ; sns.boxplot(x='Attrition',y='TrainingTimesLastYear',data=data)

plt.subplot(235)  ; sns.boxplot(x='Attrition',y='YearsAtCompany',data=data)

plt.subplot(236)  ; sns.boxplot(x='Attrition',y='YearsInCurrentRole',data=data)
#Comparing the numeric fields agains Attrition using boxplots

plt.figure(figsize=(24,6))

plt.subplot(121)  ; sns.boxplot(x='Attrition',y='YearsSinceLastPromotion',data=data)

plt.subplot(122)  ; sns.boxplot(x='Attrition',y='YearsWithCurrManager',data=data)
#Correlation plot to find interelationship of the features

plt.figure(figsize=(20, 20))

sns.heatmap(data.corr(), annot=True)
#sns.pairplot(data['BusinessTravel','Gender','Attrition'], hue='Attrition')

#sns.pairplot(data, vars=["Gender", "Attrition"])