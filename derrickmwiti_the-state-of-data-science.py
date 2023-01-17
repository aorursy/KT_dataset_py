#Import the usual suspects 

import numpy as np 

import pandas as pd #for data wrangling 

import matplotlib.pyplot as plt #for visualization

import seaborn as sns #for visualization

%matplotlib inline 
ff = pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1")
mc = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")
mc.info()
plt.title('Gender Count Plot')

sns.countplot(data=mc, y='GenderSelect')
plt.figure(figsize=(12,6))

plt.title('Employment Status Count Plot')

sns.countplot(data=mc, y='EmploymentStatus')
plt.title('Student Status Count Plot')

sns.countplot(data=mc, y='StudentStatus')
plt.figure(figsize=(12,6))

plt.title('Current Job Title Select Count Plot')

sns.countplot(data=mc, y='CurrentJobTitleSelect')
plt.figure(figsize=(12,6))

plt.title('FormalEducation Count Plot')

sns.countplot(data=mc, y='FormalEducation')
plt.figure(figsize=(12,6))

plt.title('Major Selected Count Plot')

sns.countplot(data=mc, y='MajorSelect')
age = mc['Age']

age.hist(bins=60)
kenya = mc[mc['Country']=='Kenya']
kenya[kenya['CurrentJobTitleSelect']=='Data Scientist'].info()
c = mc.sort_values('Country')['Country'].value_counts().reset_index().head(10)

c.columns = ['Country', 'Number of respondents']

c
plt.figure(figsize=(12,6))

plt.title('Learning Platform Usefulness College Count Plot')

sns.countplot(data=mc, y='LearningPlatformUsefulnessCollege')
plt.figure(figsize=(12,6))

plt.title('ProveKnowledgeSelect Count Plot')

sns.countplot(data=mc, y='ProveKnowledgeSelect')
plt.figure(figsize=(12,6))

plt.title('First Training Select Count Plot')

sns.countplot(data=mc, y='FirstTrainingSelect')
plt.figure(figsize=(12,6))

plt.title('Language Recommendation SelectCount Plot')

sns.countplot(data=mc, y='LanguageRecommendationSelect')


plt.figure(figsize=(12,20))

plt.title('Learning Platform Select Count Plot')

sns.countplot(data=mc, y='MLToolNextYearSelect')


plt.figure(figsize=(12,10))

plt.title('MLTool NextYear Select Count Plot')

sns.countplot(data=mc, y='MLToolNextYearSelect')


plt.figure(figsize=(12,6))

plt.title('How Long have you been coding?')

sns.countplot(data=mc, y='Tenure')


plt.figure(figsize=(12,6))

plt.title('Job Search Resource')

sns.countplot(data=mc, y='JobSearchResource')


plt.figure(figsize=(12,6))

plt.title('EmployerSearchMethod')

sns.countplot(data=mc, y='EmployerSearchMethod')
sal =mc[mc['CompensationAmount'].notnull()]

def salary(x):

    x = x.replace(',', '')

    try:

        return float(x)

    except:

        return np.nan

    
sal['CompensationAmount'] = sal['CompensationAmount'].apply(salary)
usa = sal[sal['Country']=='United States']
usa[(usa['CompensationAmount']>5000) & (sa['CompensationAmount']<1000000)]['CompensationAmount'].describe()