import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data=pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")
data.head()
data.shape
data.columns
data.corr()
data.info()
data['Employment'].value_counts()
data['MainBranch'].value_counts()
data_student=data[data['MainBranch']=='I am a student who is learning to code']
data_student_india=data_student[data_student['Country']=='India']
data_student_india.shape
data_student_india.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs'],axis=1,inplace=True)
data_student_india.drop(['WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat'],axis=1,inplace=True)
data_student_india.drop(['OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)
data_student_india_age=data_student_india[data_student_india['Age']>22]
data_student_india_age.head()
data_student_india_age['UndergradMajor'].value_counts().plot(kind='bar')
ct=pd.crosstab(data_student_india_age['SOFindAnswer'],data_student_india_age['SOTimeSaved'])
sns.heatmap(ct)
ct.plot(kind='bar')
sns.barplot(x='Age',y='SocialMedia',data=data_student_india)
sns.barplot(x='Age',y='EdLevel',data=data_student_india)
sns.pairplot(data_student_india_age)
sns.distplot(data_student_india_age['Age'],bins=30)
data_student_india_age.columns
sns.barplot(x='Gender',y='Age',data=data_student_india_age,estimator=np.std)
sns.violinplot(x='YearsCode',y='Age',data=data_student_india_age,split=True)
sns.stripplot(x='BlockchainIs',y='Age',data=data_student_india_age,jitter=True)
sns.barplot(x='Age',y='YearsCode',data=data_student_india_age)
sns.barplot(x='Age',y='Ethnicity',data=data_student_india_age)
sns.heatmap(pd.crosstab(data_student_india_age['DatabaseWorkedWith'],data_student_india_age['Ethnicity']))
sns.barplot(x='Age',y='ITperson',data=data_student_india_age)
data_student_india_age['Ethnicity'].value_counts().plot(kind='pie',autopct='%0.2f')