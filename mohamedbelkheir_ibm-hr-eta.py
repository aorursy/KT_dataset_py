import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
df=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.dtypes
df.isnull().any()
front=df['Attrition']
df.drop(labels=['Attrition'],axis=1,inplace=True)
df.insert(0,'Attrition',front)
df.head()
#Delete Unwanted Records
df.drop(labels=['EmployeeCount','EmployeeNumber','StockOptionLevel','StandardHours'],axis=1,inplace=True)
df.head()
#df['Gender']=df['Gender'].map({'Male':0,'Female':1}) Map doesnt work
Attrition={'Yes':1,'No':0}
df.Attrition=[Attrition[item] for item in df.Attrition]
#Get categorical values of column 
df.EducationField.unique()
# creating a dict file 
Gender={'Male':1,'Female':0}
# traversing through dataframe Gender column and writing values where key matches
df.Gender=[Gender[item] for item in df.Gender]

Field={'Life Sciences':2,'Medical':1,'Other':0,'Marketing':3,'Technical Degree':4,'Human Resources':5}
df.EducationField=[Field[item] for item in df.EducationField]
#Summary based on Attrition
df1=df.groupby('Attrition')
df1.mean()
corr=df.corr()
corr=(corr)
plt.figure(figsize=(10, 10))
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values,cmap='Blues')
corr

Attrition_Rate=df.Attrition.value_counts()/len(df)
Attrition_Rate
sns.barplot(x='Attrition',y='MonthlyIncome',hue='Gender',data=df,color='green').set_title('Employee Income Gender Distribution')
plt.figure(figsize=(10, 10))
plt.show()
sns.barplot(x='Attrition',y='DistanceFromHome',hue='Gender',data=df,color='blue').set_title('Employee Distance Gender Distribution')
plt.show()
df['Income_Range']=pd.cut(df['MonthlyIncome'],[1000,5000,10000,15000,20000])
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y='Income_Range',hue='Attrition',data=df).set_title('Employee Salary Attrition Distribution')
plt.plot()
fig=plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Attrition']==0),'JobSatisfaction'],color='g',shade=False,label='No Attrition')
ax=sns.kdeplot(df.loc[(df['Attrition']==1),'JobSatisfaction'],color='r',shade=True,label='Attrition')
ax.set(xlabel='Employee Job Satisfaction Rating',ylabel='Frequency')
plt.title('Employee Job Satisfaction Rating - Attrition vs No Attrition')
fig=plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Attrition']==0),'WorkLifeBalance'],color='g',shade=False,label='No Attrition')
ax=sns.kdeplot(df.loc[(df['Attrition']==1),'WorkLifeBalance'],color='r',shade=True,label='Attrition')
ax.set(xlabel='Employee WorkLifeBalance Rating',ylabel='Frequency')
plt.title('Employee WorkLifeBalance Rating - Attrition vs No Attrition')
fig=plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Attrition']==0),'RelationshipSatisfaction'],color='g',shade=False,label='No Attrition')
ax=sns.kdeplot(df.loc[(df['Attrition']==1),'RelationshipSatisfaction'],color='r',shade=True,label='Attrition')
ax.set(xlabel='Employee RelationshipSatisfaction Rating',ylabel='Frequency')
plt.title('Employee Relationship Satisfaction Rating - Attrition vs No Attrition')
fig=plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Attrition']==0),'YearsAtCompany'],color='g',shade=False,label='No Attrition')
ax=sns.kdeplot(df.loc[(df['Attrition']==1),'YearsAtCompany'],color='r',shade=True,label='Attrition')
ax.set(xlabel='Employee YearsAtCompany ',ylabel='Frequency')
plt.title('Employee YearsAtCompany - Attrition vs No Attrition')