import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")
data.head()
data.shape
data.info()
data['Employment'].value_counts()
data['BetterLife'].value_counts()
data['ITperson'].value_counts()
data['MainBranch'].value_counts()
data['Hobbyist'].value_counts()
data['Respondent'].value_counts()
data['UndergradMajor'].value_counts()
data['Age'].value_counts()
data['Country'].value_counts()
data['YearsCode'].value_counts()
data_student=data[data['MainBranch']=='I am a student who is learning to code']
data_student_india=data_student[data_student['Country']=='India']
data_student_US=data_student[data_student['Country']=='United States']
data_hobby=data[data['MainBranch']=='I code primarily as a hobby']
data_hobby_india=data_hobby[data_hobby['Country']=='India']
data_hobby_US=data_hobby[data_hobby['Country']=='United States']
data_student_india.shape
data_student_US.shape
data_hobby_india.shape
data_hobby_US.shape
data_hobby_india=data_hobby[data_hobby['Country']=='India']

sns.barplot(y='YearsCode',x='Age',data=data_hobby_india,color='red')

data_hobby_US=data_hobby[data_hobby['Country']=='United States']

sns.barplot(y='YearsCode',x='Age',data=data_hobby_US,color='blue')

data_student_india.columns
data_student_india.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs','WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)
data_student_US.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs','WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)
data_student_india.shape
data_student_US.shape
data_student_india_age=data_student_india[data_student_india['Age']>20]
data_student_US_age=data_student_US[data_student_US['Age']>20]
data_student_india_age['UndergradMajor'].value_counts().plot(kind='barh')
data_student_US_age['UndergradMajor'].value_counts().plot(kind='barh',color='red')
data['LanguageWorkedWith'].value_counts()
data_student_india=data_student[data_student['Country']=='India']

sns.barplot(x='Age',y='SocialMedia',data=data_student_india)

data_student_US=data_student[data_student['Country']=='United States']

sns.barplot(x='Age',y='SocialMedia',data=data_student_US)

sns.barplot(x='Age',y='ITperson',data=data_student_india_age)
sns.barplot(x='Age',y='ITperson',data=data_student_US_age)
sns.barplot(x='Age',y='Ethnicity',data=data_student_india_age)
sns.barplot(x='Age',y='Ethnicity',data=data_student_US_age)

sns.barplot(x='Age',y='EdLevel',data=data_student_india_age)
sns.barplot(x='Age',y='EdLevel',data=data_student_US_age)
sns.barplot(y='Gender',x='Age',data=data_student_india_age)
sns.barplot(y='Gender',x='Age',data=data_student_US_age)
data_student_india.columns
data_student_india_age['BetterLife'].value_counts()
sns.barplot(y='Employment',x='Age',data=data_student_india)
sns.barplot(y='Employment',x='Age',data=data_student_US)
sns.barplot(y='Dependents',x='Age',data=data_student_india_age)
sns.barplot(y='Dependents',x='Age',data=data_student_US_age)
data_student_india_age['BetterLife'].value_counts().plot(kind='pie',autopct='%0.2f')
data_student_US_age['BetterLife'].value_counts().plot(kind='pie',autopct='%0.2f')