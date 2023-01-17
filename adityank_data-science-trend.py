#Render Matplotlib Plots Inline

%matplotlib inline



#Import the standard Python Scientific Libraries

import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



#Suppress Deprecation and Incorrect Usage Warnings 

import warnings

warnings.filterwarnings('ignore')
#Load MCQ Responses into a Pandas DataFrame

data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)
sns.countplot(data=data,y='GenderSelect')
c = data[['Country','Age']].groupby('Country').count().sort_values('Age',ascending=False).head(5)

c.reset_index(inplace=True)

plt.pie(c['Age'],labels=c['Country'])

plt.show()

data[['EmploymentStatus','Country']].groupby('EmploymentStatus').count()

sns.countplot(data=data,y='EmploymentStatus')
age = data[['EmploymentStatus','Age']].groupby('Age').count().sort_values('EmploymentStatus',ascending=False).head(10)

age.reset_index(inplace=True)

age=age.rename(columns = {'EmploymentStatus':'No Of People'})

print (age)

q = data[((data['Country'] == 'India') | (data['Country'] == 'United States')) & (data['Age'] >= 20) & (data['Age'] <= 35)][['Age','Country']].groupby('Country').count()



q=q.rename(columns = {'Age':'No Of People'})

q.plot(kind='bar')

x = data[((data['Country'] == 'India') | (data['Country'] == 'United States')) & (data['Age'] >= 20) & (data['Age'] <= 35)]

plt.figure(figsize=(8, 10))

sns.countplot(y="Age", hue="Country", data=x)
x = data[((data['Country'] == 'India') | (data['Country'] == 'United States')) & ((data['Age'] >= 20) & (data['Age'] <= 35)) & (data['EmploymentStatus'] == 'Not employed, but looking for work')]

plt.figure(figsize=(8, 10))

sns.countplot(y="Age", hue="Country", data=x)
lang = data[['LanguageRecommendationSelect','Country']]

lang.reset_index(inplace=True)

lang=lang.rename(columns = {'Country':'Recommendations'})

plt.figure(figsize=(10,8))

sns.countplot(data=lang,x='LanguageRecommendationSelect')
l = data[(data['GenderSelect'].notnull()) & ((data['LanguageRecommendationSelect'] == 'Python') | (data['LanguageRecommendationSelect'] == 'R'))]

plt.figure(figsize=(8,10))

sns.countplot(y="GenderSelect", hue="LanguageRecommendationSelect", data=l)
l = data[(data['Country'].notnull()) & ((data['LanguageRecommendationSelect'] == 'Python') | (data['LanguageRecommendationSelect'] == 'R')) & ((data['Age'] >= 20) & (data['Age'] <= 35))]

plt.figure(figsize=(12,15))

sns.countplot(y="Country", hue="LanguageRecommendationSelect", data=l)
d = data[(data['CurrentJobTitleSelect'].notnull()) & ((data['LanguageRecommendationSelect'] == 'Python') | (data['LanguageRecommendationSelect'] == 'R'))]

plt.figure(figsize=(8,10))

sns.countplot(y="CurrentJobTitleSelect", hue="LanguageRecommendationSelect", data=d)

tool = data[['MLToolNextYearSelect','Country']].groupby('MLToolNextYearSelect').count().sort_values('Country',ascending=False).head(5)

tool.reset_index(inplace=True)

tool=tool.rename(columns = {'Country':'Recommendations'})

tool.plot(kind='bar',x='MLToolNextYearSelect',figsize=(10,8))
method = data[['MLMethodNextYearSelect','Country']].groupby('MLMethodNextYearSelect').count().sort_values('Country',ascending=False).head(5)

method.reset_index(inplace=True)

method=method.rename(columns = {'Country':'Recommendations'})

method.plot(kind='bar',x='MLMethodNextYearSelect',figsize=(10,8))
coder = data[['CodeWriter','GenderSelect']]

sns.countplot(data=coder,x='CodeWriter')
codergender = data[((data['GenderSelect'] == 'Male') | (data['GenderSelect'] == 'Female')) & (data['CodeWriter'] == 'Yes')][['CodeWriter','GenderSelect']].groupby('GenderSelect').count()

codergender.plot(kind='bar')
hrs = data[['TimeSpentStudying','Country']]

sns.countplot(data=hrs,x='TimeSpentStudying')
hl = data[(data['TimeSpentStudying'].notnull()) & ((data['LanguageRecommendationSelect'] == 'Python') | (data['LanguageRecommendationSelect'] == 'R'))]

plt.figure(figsize=(10,8))

sns.countplot(x="TimeSpentStudying", hue="LanguageRecommendationSelect", data=hl)

hl = data[(data['TimeSpentStudying'].notnull()) & ((data['Country'] == 'India') | (data['Country'] == 'United States'))]

plt.figure(figsize=(10,8))

sns.countplot(x="TimeSpentStudying", hue="Country", data=hl)

js = data[['JobSkillImportanceBigData','JobSkillImportanceSQL','JobSkillImportanceVisualizations']]

sns.countplot(data=js,x='JobSkillImportanceVisualizations')