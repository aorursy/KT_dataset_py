import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data=pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')

data1=pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')
data.head()
data1.head()
data.shape
data.info
data.columns
data.corr()
data.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs','WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)
data
data['Employment'].value_counts()
data['Employment'].value_counts().plot(kind='bar')
data['MainBranch'].value_counts()
data['MainBranch'].value_counts().plot(kind='pie', autopct='%0.2f')
data['Country'].value_counts().head(10)
data['Country'].value_counts().head(10).plot(kind='bar')
data['Age'].value_counts().head(10).plot(kind='bar')
sns.barplot(x='Age',y= 'Employment',data=data)
data['UndergradMajor'].value_counts().head(10).plot(kind='bar')
data['SocialMedia'].value_counts().head(10)
data['SocialMedia'].value_counts().head(10).plot(kind='bar')
sns.barplot(x='Age',y= 'SocialMedia',data=data)
data['EdLevel'].value_counts()
data['EdLevel'].value_counts().plot(kind='bar')
sns.barplot(x='Age',y= 'EdLevel',data=data)